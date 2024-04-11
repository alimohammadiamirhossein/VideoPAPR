import yaml
import argparse
import torch
import os
import shutil
import zipfile
import matplotlib.pyplot as plt
import numpy as np
import cv2
import random
import copy
import bisect
import time
import sys
import io
import imageio
from PIL import Image
from logger import *
from dataset import get_dataset, get_loader
from models import get_model, get_loss
from models.sjc.adapt_sd import StableDiffusion, SD
from models.sjc.adapt import ScoreAdapter
from models.model import PAPR
from models.sjc.my3d import get_T
from models.sjc.pose import Poser, PoseConfig
from dataset.dataset import RINDataset
from dataset.utils import get_rays
from torchvision import transforms


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class DictAsMember(dict):
    def __getattr__(self, name):
        value = self[name]
        if isinstance(value, dict):
            value = DictAsMember(value)
        return value

def gradient_x(img: torch.Tensor) -> torch.Tensor:
    return img[:, :-1] - img[:, 1:]

def gradient_y(img: torch.Tensor) -> torch.Tensor:
    return img[:-1, :] - img[1:, :]

def depth_smooth_loss(depth):
    grad_x, grad_y = gradient_x(depth), gradient_y(depth)
    return (grad_x.abs().mean() + grad_y.abs().mean()) / 2.

def setup_seed(seed):
    torch.manual_seed(seed)
    if DEVICE == 'cuda:0':
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def parse_args():
    parser = argparse.ArgumentParser(description="PAPR")
    parser.add_argument('--opt', type=str, default="configs/nerfsyn/chair.yml", help='Option file path')
    parser.add_argument('--resume', type=int, default=0, help='Resume training')
    return parser.parse_args()


def eval_step(steps, model, device, dataset, eval_dataset, batch, loss_fn, train_out, args, train_losses, eval_losses, eval_psnrs, pt_lrs, tx_lrs):
    step = steps[-1]
    train_img_idx, _, train_patch, _, _  = batch
    train_img, train_rayd, train_rayo = dataset.get_full_img(train_img_idx[0])
    img, rayd, rayo = eval_dataset.get_full_img(args.eval.img_idx)
    c2w = dataset.get_c2w(args.eval.img_idx)
    
    N, H, W, _ = rayd.shape
    num_pts, _ = model.points.shape

    rayo = rayo.to(DEVICE)
    rayd = rayd.to(DEVICE)
    img = img.to(DEVICE)
    c2w = c2w.to(DEVICE)

    topk = min([num_pts, model.select_k])

    selected_points = torch.zeros(1, H, W, topk, 3)

    bkg_seq_len_attn = 0
    tx_opt = args.models.transformer
    feat_dim = tx_opt.embed.d_ff_out if tx_opt.embed.share_embed else tx_opt.embed.value.d_ff_out
    if model.bkg_feats is not None:
        bkg_seq_len_attn = model.bkg_feats.shape[0]
    feature_map = torch.zeros(N, H, W, 1, feat_dim).to(DEVICE)
    attn = torch.zeros(N, H, W, topk + bkg_seq_len_attn, 1).to(DEVICE)

    with torch.no_grad():
        for height_start in range(0, H, args.eval.max_height):
            for width_start in range(0, W, args.eval.max_width):
                height_end = min(height_start + args.eval.max_height, H)
                width_end = min(width_start + args.eval.max_width, W)

                feature_map[:, height_start:height_end, width_start:width_end, :, :], \
                attn[:, height_start:height_end, width_start:width_end, :, :] = model.evaluate(rayo, rayd[:, height_start:height_end, width_start:width_end], c2w, step=step)

                selected_points[:, height_start:height_end, width_start:width_end, :, :] = model.selected_points
        
        if args.models.use_renderer:
            foreground_rgb = model.renderer(feature_map.squeeze(-2).permute(0, 3, 1, 2)).permute(0, 2, 3, 1).unsqueeze(-2)   # (N, H, W, 1, 3)
            if model.bkg_feats is not None:
                bkg_attn = attn[..., topk:, :]
                if args.models.normalize_topk_attn:
                    rgb = foreground_rgb * (1 - bkg_attn) + model.bkg_feats.expand(N, H, W, -1, -1) * bkg_attn
                else:
                    rgb = foreground_rgb + model.bkg_feats.expand(N, H, W, -1, -1) * bkg_attn
                rgb = rgb.squeeze(-2)
            else:
                rgb = foreground_rgb.squeeze(-2)
        else:
            rgb = feature_map.squeeze(-2)
                
        rgb = model.last_act(rgb)
        rgb = torch.clamp(rgb, 0, 1)

        eval_loss = loss_fn(rgb, img)
        eval_psnr = -10. * np.log(((rgb - img)**2).mean().item()) / np.log(10.)

        model.clear_grad()

    eval_losses.append(eval_loss.item())
    eval_psnrs.append(eval_psnr.item())

    print("Eval step:", step, "train_loss:", train_losses[-1], "eval_loss:", eval_losses[-1], "eval_psnr:", eval_psnrs[-1])

    log_dir = os.path.join(args.save_dir, args.index)
    os.makedirs(log_dir, exist_ok=True)
    if args.eval.save_fig:
        os.makedirs(os.path.join(log_dir, "train_main_plots"), exist_ok=True)
        os.makedirs(os.path.join(log_dir, "train_pcd_plots"), exist_ok=True)

        coord_scale = args.dataset.coord_scale
        pt_plot_scale = 1.0 * coord_scale
        if "Barn" in args.dataset.path:
            pt_plot_scale *= 1.8
        if "Family" in args.dataset.path:
            pt_plot_scale *= 0.5    

        # calculate depth, weighted sum the distances from top K points to image plane
        od = -rayo
        D = torch.sum(od * rayo)
        dists = torch.abs(torch.sum(selected_points.to(DEVICE) * od, -1) - D) / torch.norm(od)
        if model.bkg_feats is not None:
            dists = torch.cat([dists, torch.ones(N, H, W, model.bkg_feats.shape[0]).to(DEVICE) * 0], dim=-1)
        cur_depth = (torch.sum(attn.squeeze(-1).to(DEVICE) * dists, dim=-1)).detach().cpu()

        train_tgt_rgb = train_img.squeeze().cpu().numpy().astype(np.float32)
        train_tgt_patch = train_patch[0].cpu().numpy().astype(np.float32)
        train_pred_patch = train_out[0]
        test_tgt_rgb = img.squeeze().cpu().numpy().astype(np.float32)
        test_pred_rgb = rgb.squeeze().detach().cpu().numpy().astype(np.float32)
        points_np = model.points.detach().cpu().numpy()
        depth = cur_depth.squeeze().numpy().astype(np.float32)
        points_influ_scores_np = None
        if model.points_influ_scores is not None:
            points_influ_scores_np = model.points_influ_scores.squeeze().detach().cpu().numpy()

        # main plot
        main_plot = get_training_main_plot(args.index, steps, train_tgt_rgb, train_tgt_patch, train_pred_patch, test_tgt_rgb, test_pred_rgb, train_losses, 
                                           eval_losses, points_np, pt_plot_scale, depth, pt_lrs, tx_lrs, eval_psnrs, points_influ_scores_np)
        save_name = os.path.join(log_dir, "train_main_plots", "%s_iter_%d.png" % (args.index, step))
        main_plot.save(save_name)

        # point cloud plot
        ro = train_rayo.squeeze().detach().cpu().numpy()
        rd = train_rayd.squeeze().detach().cpu().numpy()
        
        pcd_plot = get_training_pcd_plot(args.index, steps[-1], ro, rd, points_np, args.dataset.coord_scale, pt_plot_scale, points_influ_scores_np)
        save_name = os.path.join(log_dir, "train_pcd_plots", "%s_iter_%d.png" % (args.index, step))
        pcd_plot.save(save_name)

    model.save(step, log_dir)
    if step % 50000 == 0:
        torch.save(model.state_dict(), os.path.join(log_dir, "model_%d.pth" % step))

    torch.save(torch.tensor(train_losses), os.path.join(log_dir, "train_losses.pth"))
    torch.save(torch.tensor(eval_losses), os.path.join(log_dir, "eval_losses.pth"))
    torch.save(torch.tensor(eval_psnrs), os.path.join(log_dir, "eval_psnrs.pth"))

    return 0
from dataset.utils import extract_patches
from einops import rearrange
from models.sjc.pose import PoseConfig, camera_pose, sample_near_eye


def backward_sjc__near_loss(
        sd_model: StableDiffusion,
        papr_model: PAPR,
        poser: Poser,
        input_image,
        input_pose,
        dataset: RINDataset,
        step,
        img_idx,
        var_red: bool=True,
        n_steps:    int = 10000,
        args=None,
        depth_smooth_weight=1e5,
        near_view_weight = 1e5,
        emptiness_step:     float = 0.5):

    Ks, poses, prompt_prefixes = poser.sample_train(n_steps)

    target_pose = torch.Tensor(poses[step])

    rays_o, rays_d = get_rays(input_image.shape[1], input_image.shape[2], dataset.focal_x, dataset.focal_y, target_pose.unsqueeze(0))

    input_pose = input_pose.cpu().numpy()



    input_im = input_image.cpu().numpy()
    
    input_pose[:3, -1] = input_pose[:3, -1] / np.linalg.norm(input_pose[:3, -1]) * poser.R
    
    tforms1 = transforms.Compose([transforms.Resize(32), transforms.CenterCrop((32, 32))])


    # Step 2: Reshape (if necessary) and select the first image in the batch
    # Convert from (batch, channels, height, width) to (height, width, channels)
    input_im = input_im[0]

    input_im = cv2.resize(input_im, dsize=(256, 256), interpolation=cv2.INTER_CUBIC)


    input_im = torch.as_tensor(input_im, dtype=float, device=DEVICE)
    input_im = input_im.permute(2, 0, 1)[None, :, :]
    input_im = input_im * 2. - 1.


    with torch.no_grad():

            tforms = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop((256, 256))
            ])

            input_im = tforms(input_im)
            #print('input_im after transformation: ', input_im.shape, flush=True)
            #input_im = input_im.squeeze(0)
            #print('input_im after unsqueeze: ', input_im.shape, flush=True)
            # get input embedding
            sd_model.clip_emb = sd_model.model.get_learned_conditioning(input_im.float()).tile(1,1,1).detach()
            sd_model.vae_emb = sd_model.model.encode_first_stage(input_im.float()).mode().detach()

    # target_pose = np.expand_dims(target_pose, axis=0)
    # print('pose: ', target_pose)
    
    
    #print('rays are generated')
    
    T_target = target_pose[:3, -1]
    T_cond = input_pose[:3, -1]
    T = get_T(T_target, T_cond)
    T = T.to(DEVICE)

    # print('T.shape: ', T.shape)
    # print('T is generated')
    
    bs = 1
    ts = sd_model.us[30:-10]

    #print('ts is generated')

    papr_rgb_out = papr_model(rays_o.to(DEVICE), rays_d.to(DEVICE), target_pose.to(DEVICE), step)
    papr_rgb_out = tforms1(rearrange(papr_rgb_out, "1 h w c -> 1 c h w"))
    #print('papr_rgb_out.shapp#: ', papr_rgb_out.shape)
    papr_rgb_out = torch.cat((papr_rgb_out, torch.zeros((1, 1, 32, 32), requires_grad=True).to(DEVICE)), dim=1)
    #print('papr_rgb_out.shap: ######## ', papr_rgb_out.shape)
    #print('PAPR output is generated in SJC')
    # papr_rgb_out = torch.zeros((1, 4, 32, 32))

    with torch.no_grad():
        chosen_σs = np.random.choice(ts, bs, replace=False)
        chosen_σs = chosen_σs.reshape(-1, 1, 1, 1)
        chosen_σs = torch.as_tensor(chosen_σs, device=DEVICE, dtype=torch.float32)

        # TODO we should sample from a different angle than traget image
        noise = torch.randn(bs, *papr_rgb_out.shape[1:], device=DEVICE)

        zs = papr_rgb_out + chosen_σs * noise
        
        # TODO For now traget and input images are the same
        # TODO what should we do about T and different angels? rayd, rayo should be calculated from pose and veiwpoint 
        score_conds = sd_model.img_emb(input_im, conditioning_key='hybrid', T=T)

        # print('zs.shape: ', zs.shape)
        # print('chosen_σs.shape: ', chosen_σs.shape)

        Ds = sd_model.denoise_objaverse(zs, chosen_σs, score_conds)

        if var_red:
            grad = (Ds - papr_rgb_out) / chosen_σs
        else:
            grad = (Ds - zs) / chosen_σs
        
        grad = grad.mean(0, keepdim=True)
    
    # print('SJC loss is ready')    
    papr_rgb_out.backward(-grad, retain_graph=True)


    # depth smoothness loss
    in_depth, _ = get_depth(step, rays_d.to(DEVICE), rays_o.to(DEVICE), papr_model, input_pose)
    input_smooth_loss = depth_smooth_loss(in_depth) * depth_smooth_weight * 0.1
    input_smooth_loss.backward(retain_graph=True)

    # near view loss
    near_loss, depth = backward_near_loss(step, rays_d.to(DEVICE), rays_o.to(DEVICE), papr_model, target_pose, poser, near_view_weight)

    smooth_loss = depth_smooth_loss(depth) * depth_smooth_weight
    if step >= emptiness_step * args.training.steps:
        smooth_loss.backward(retain_graph=True)

    return near_loss, input_smooth_loss, smooth_loss

def backward_near_loss(step, rayd, rayo, papr_model, target_pose, poser, near_view_weight):
    depth, rgb = get_depth(step, rayd, rayo, papr_model, target_pose)
    # print('1st depth')
    
    eye = target_pose[:3, -1].cpu().detach().numpy()
    near_eye = sample_near_eye(eye)
    # print(near_eye)
    # print(poser.up)
    near_pose = camera_pose(near_eye, -near_eye, poser.up)
    near_pose = torch.from_numpy(near_pose).to(DEVICE)
    depth_near, rgb_near, = get_depth(step, rayd, rayo, papr_model, near_pose)
    # print('2nd depth')
    near_loss = ((rgb_near - rgb).abs().mean() + (depth_near - depth).abs().mean()) * near_view_weight
    near_loss.backward(retain_graph=True)
    #print(near_loss)
    return near_loss, depth

    # print('Near loss is backwarded')


def train_step(step, papr_model, device, dataset, batch, loss_fn, args, sd_model, poser, view_weight=10000):
    img_idx, _, tgt, rayd, rayo = batch
    
    # TODO c2w is pose and focals are angels
    c2w = dataset.get_c2w(img_idx[0])


    rayo = rayo.to(DEVICE)
    rayd = rayd.to(DEVICE)
    tgt = tgt.to(DEVICE)
    c2w = c2w.to(DEVICE)

    papr_model.clear_grad()
    # TODO

    near_loss, input_smooth_loss, smooth_loss = backward_sjc__near_loss(sd_model, papr_model, poser, tgt, c2w, dataset, step, img_idx, args=args)


    out = papr_model(rayo, rayd, c2w, step)
    out = papr_model.last_act(out)

    loss = loss_fn(out, tgt) * float(view_weight)

    papr_model.scaler.scale(loss).backward()
    papr_model.step(step)
    if args.scaler_min_scale > 0 and papr_model.scaler.get_scale() < args.scaler_min_scale:
        papr_model.scaler.update(args.scaler_min_scale)
    else:
        papr_model.scaler.update()

    loss = loss + near_loss

    return loss.detach().item(), near_loss.detach().item(), out.detach().cpu().numpy()
    

def get_depth(step, rayd, rayo, papr_model, c2w):
    

    N, H, W, _ = rayd.shape
    num_pts, _ = papr_model.points.shape


    topk = min([num_pts, papr_model.select_k])

    feature_map, attn = papr_model.evaluate(rayo.to(DEVICE), rayd.to(DEVICE), c2w.to(DEVICE), step=step)
    
    rgb = papr_model(rayo.to(DEVICE), rayd.to(DEVICE), c2w.to(DEVICE), step)

    selected_points = papr_model.selected_points

    od = -rayo
    D = torch.sum(od * rayo)
    dists = torch.abs(torch.sum(selected_points.to(DEVICE) * od, -1) - D) / torch.norm(od)
    if papr_model.bkg_feats is not None:
        dists = torch.cat([dists, torch.ones(N, H, W, papr_model.bkg_feats.shape[0]).to(DEVICE) * 0], dim=-1)
    cur_depth = (torch.sum(attn.squeeze(-1).to(DEVICE) * dists, dim=-1)).detach().cpu()
    depth = cur_depth.squeeze()
    
    return depth, rgb
    

def train_and_eval(start_step, model, device, dataset, eval_dataset, losses, args, sd_model, poser):

    trainloader = get_loader(dataset, args.dataset, mode="train")

    loss_fn = get_loss(args.training.losses)
    loss_fn = loss_fn.to(DEVICE)

    log_dir = os.path.join(args.save_dir, args.index)
    os.makedirs(os.path.join(log_dir, "test"), exist_ok=True)
    log_dir = os.path.join(log_dir, "test")

    steps = []
    train_losses, eval_losses, eval_psnrs = losses
    pt_lrs = []
    tx_lrs = []

    avg_train_loss = 0.
    avg_near_loss = 0.
    step = start_step
    eval_step_cnt = start_step
    pruned = False
    pc_frames = []

    print("Start step:", start_step, "Total steps:", args.training.steps)
    start_time = time.time()
    while step < args.training.steps:
        for _, batch in enumerate(trainloader):
            if (args.training.prune_steps > 0) and (step < args.training.prune_stop) and (step >= args.training.prune_start):
                if len(args.training.prune_steps_list) > 0 and step % args.training.prune_steps == 0:
                    cur_prune_thresh = args.training.prune_thresh_list[bisect.bisect_left(args.training.prune_steps_list, step)]
                    model.clear_optimizer()
                    model.clear_scheduler()
                    num_pruned = model.prune_points(cur_prune_thresh)
                    model.init_optimizers(step)
                    pruned = True
                    print("Step %d: Pruned %d points, prune threshold %f" % (step, num_pruned, cur_prune_thresh))

                elif step % args.training.prune_steps == 0:
                    model.clear_optimizer()
                    model.clear_scheduler()
                    num_pruned = model.prune_points(args.training.prune_thresh)
                    model.init_optimizers(step)
                    pruned = True
                    print("Step %d: Pruned %d points" % (step, num_pruned))

            if pruned and len(args.training.add_steps_list) > 0:
                if step in args.training.add_steps_list:
                    cur_add_num = args.training.add_num_list[args.training.add_steps_list.index(step)]
                    if 'max_num_pts' in args and args.max_num_pts > 0:
                        cur_add_num = min(cur_add_num, args.max_num_pts - model.points.shape[0])
                    
                    if cur_add_num > 0:
                        model.clear_optimizer()
                        model.clear_scheduler()
                        num_added = model.add_points(cur_add_num)
                        model.init_optimizers(step)
                        model.added_points = True
                        print("Step %d: Added %d points" % (step, num_added))

            elif pruned and (args.training.add_steps > 0) and (step % args.training.add_steps == 0) and (step < args.training.add_stop) and (step >= args.training.add_start):
                cur_add_num = args.training.add_num
                if 'max_num_pts' in args and args.max_num_pts > 0:
                    cur_add_num = min(cur_add_num, args.max_num_pts - model.points.shape[0])
                
                if cur_add_num > 0:
                    model.clear_optimizer()
                    model.clear_scheduler()
                    num_added = model.add_points(args.training.add_num)
                    model.init_optimizers(step)
                    model.added_points = True
                    print("Step %d: Added %d points" % (step, num_added))

            loss, near_loss, out = train_step(step, model, DEVICE, dataset, batch, loss_fn, args, sd_model, poser)
            torch.cuda.empty_cache()
            avg_train_loss += loss
            step += 1
            eval_step_cnt += 1
            
            if step % 200 == 0:
                time_used = time.time() - start_time
                print("Train step:", step, "loss:", loss, "near_loss:", near_loss, "tx_lr:", model.tx_lr, "pts_lr:", model.pts_lr, "scale:", model.scaler.get_scale(), f"time: {time_used:.2f}s")
                start_time = time.time()

            if (step % args.eval.step == 0) or (step % 500 == 0 and step < 10000):
                train_losses.append(avg_train_loss / eval_step_cnt)
                pt_lrs.append(model.pts_lr)
                tx_lrs.append(model.tx_lr)
                steps.append(step)
                eval_step(steps, model, DEVICE, dataset, eval_dataset, batch, loss_fn, out, args, train_losses, eval_losses, eval_psnrs, pt_lrs, tx_lrs)
                avg_train_loss = 0.
                eval_step_cnt = 0

            if ((step - 1) % 200 == 0) and args.eval.save_fig:
                coord_scale = args.dataset.coord_scale
                pt_plot_scale = 0.8 * coord_scale
                if "Barn" in args.dataset.path:
                    pt_plot_scale *= 1.5
                if "Family" in args.dataset.path:
                    pt_plot_scale *= 0.5    

                pc_dir = os.path.join(log_dir, "point_clouds")
                os.makedirs(pc_dir, exist_ok=True)

                points_np = model.points.detach().cpu().numpy()
                points_influ_scores_np = None
                if model.points_influ_scores is not None:
                    points_influ_scores_np = model.points_influ_scores.squeeze().detach().cpu().numpy()
                pcd_plot = get_training_pcd_single_plot(step, points_np, pt_plot_scale, points_influ_scores_np)
                pc_frames.append(pcd_plot)
                
                if step == 1:
                    pcd_plot.save(os.path.join(pc_dir, "init_pcd.png"))

            if step >= args.training.steps:
                break

    if args.eval.save_fig and pc_frames != []:
        f = os.path.join(log_dir, f"{args.index}-pc.mp4")
        imageio.mimwrite(f, pc_frames, fps=30, quality=10)

    print("Training finished!")

            
def main(args, eval_args, resume):
    log_dir = os.path.join(args.save_dir, args.index)

    model = get_model(args, DEVICE)

    sd_model = SD(variant="objaverse", scale=100.0, precision='full')
    sd_model = sd_model.make()
    print('SD is loaded')
    poser = PoseConfig(rend_hw=32, FoV=49.1, R=2.0)
    print('Poser is loaded')
    poser = poser.make()



    dataset = get_dataset(args.dataset, mode="train")
    eval_dataset = get_dataset(eval_args.dataset, mode="test")
    model = model.to(DEVICE)

    start_step = 0
    losses = [[], [], []]
    if resume > 0:
        start_step = model.load(log_dir)

        train_losses = torch.load(os.path.join(log_dir, "train_losses.pth")).tolist()
        eval_losses = torch.load(os.path.join(log_dir, "eval_losses.pth")).tolist()
        eval_psnrs = torch.load(os.path.join(log_dir, "eval_psnrs.pth")).tolist()
        losses = [train_losses, eval_losses, eval_psnrs]

        print("!!!!! Resume from step %s" % start_step)
    elif args.load_path:
        try:
            resume_step = model.load(os.path.join(args.save_dir, args.load_path))
        except:
            model_state_dict = torch.load(os.path.join(args.save_dir, args.load_path, "model.pth"))
            for step, state_dict in model_state_dict.items():
                resume_step = step
                model.load_my_state_dict(state_dict)
        print("!!!!! Loaded model from %s at step %s" % (args.load_path, resume_step))

    train_and_eval(start_step, model, DEVICE, dataset, eval_dataset, losses, args, sd_model, poser=poser)
    
    if DEVICE == 'cuda':
        print(torch.cuda.memory_summary())


if __name__ == '__main__':

    args = parse_args()
    with open(args.opt, 'r') as f:
        config = yaml.safe_load(f)
    eval_config = copy.deepcopy(config)
    eval_config['dataset'].update(eval_config['eval']['dataset'])
    eval_config = DictAsMember(eval_config)
    config = DictAsMember(config)

    log_dir = os.path.join(config.save_dir, config.index)
    os.makedirs(log_dir, exist_ok=True)

    sys.stdout = Logger(os.path.join(log_dir, 'train.log'), sys.stdout)
    sys.stderr = Logger(os.path.join(log_dir, 'train_error.log'), sys.stderr)

    shutil.copyfile(__file__, os.path.join(log_dir, os.path.basename(__file__)))
    shutil.copyfile(args.opt, os.path.join(log_dir, os.path.basename(args.opt)))

    find_all_python_files_and_zip(".", os.path.join(log_dir, "code.zip"))

    setup_seed(config.seed)

    main(config, eval_config, args.resume)
