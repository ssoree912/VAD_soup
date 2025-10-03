import torch

import os
import numpy as np
import yaml
import argparse

from utils import *
from model import AD_Model, Memory_module
from loss import Loss_bce
from data.dataset_loader import CreateDataset

def train(dataloader, model, optimizer_model, criterion, epoch, device):
    with torch.set_grad_enabled(True):
        model.train()

        total_loss = 0.0
        total_samples = 0
        total_batches = 0

        for features, pseudo_labels, reweight, _, _ in dataloader:
            bs, nc, t, dim = features.shape
            features = features.type(torch.float).to(device)
            pseudo_labels = pseudo_labels.type(torch.float).to(device)
            reweight = reweight.type(torch.float).to(device)
            scores = model(features)
            loss_cls = criterion(scores, pseudo_labels, reweight)

            optimizer_model.zero_grad()
            loss_cls.backward()
            optimizer_model.step()

            total_loss += loss_cls.item()
            total_samples += bs * nc * t
            total_batches += 1

        avg_loss = total_loss / max(total_batches, 1)

        logger.info('Epoch: [{:.0f}/{:.0f}], '
                    'sample_num: {}, '
                    'loss_cls: {:.4f}.'.format(epoch, args.epochs, total_samples, avg_loss))

        return avg_loss

def test(model, test_loader, device, is_train_sample=False):
    if is_train_sample:
        with torch.no_grad():
            model.eval()
            losses_dict={}
            for features, label_frames, video_name in test_loader:
                features = features.type(torch.float).to(device)
                label_frames = label_frames.type(torch.float).to(device)
                outputs = model(features)
                scores = outputs.squeeze().cpu().numpy()
        
                losses_dict[video_name[0]] = scores

        logger.info('Eval abnormal videos in training set. Finished!')

        return losses_dict
    else:
        with torch.no_grad():
            model.eval()
            total_scores = []
            total_labels = []
            scores_dist = {}
            for features, label_frames, video_name in test_loader:
                features = features.type(torch.float).to(device)
                label_frames = label_frames.type(torch.float).to(device)
                outputs = model(features)

                scores = outputs.squeeze().cpu().numpy()
                scores_dist[video_name[0]] = scores

                for score, label in zip(scores, label_frames[0]):
                    score = [score] * args.segment_len
                    label = label.detach().cpu().numpy().astype(int).tolist()
                    total_scores.extend(score)
                    total_labels.extend(label)

        total_score_frames = np.array(total_scores)
        total_label_frames = np.array(total_labels)

        prauc_frames, rocauc_frames = calc_metrics(total_score_frames, total_label_frames)
    
        logger.info('Testing: pr@ {:.2f}%, '
              'auc@ {:.2f}% \t'.format(prauc_frames, rocauc_frames))
        
        return scores_dist, prauc_frames, rocauc_frames

def prepare_log_files(args):
    param_str = '{}_lr_{}_{}'.format(args.dataset, args.lr, get_timestamp())
    
    ckpt_path = os.path.join(args.ckpt_path, args.dataset, param_str)
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)

    logger_path = os.path.join(args.logger_path, args.dataset)
    if not os.path.exists(logger_path):
        os.makedirs(logger_path)

    logger = get_logger(logger_path+'/{}.txt'.format(param_str))
    logger.info('Train this model at time {}'.format(get_timestamp()))
    log_param(logger, args)
    logger.info(param_str)

    return logger, ckpt_path

def prepare_model(args, device):
    cls_model = AD_Model(args.feature_dim, 512, args.dropout_rate)

    if torch.cuda.is_available():
        cls_model.to(device)
        torch.backends.cudnn.benchmark = True

    return cls_model

def extract_features(dataloader):
    features_all = []
    video_name_all = []
    pseudo_labels_all = []
    normal_video_names_high_confidence = []
    for features, pseudo_labels, _, high_confidence_norvideo_tag, video_name in dataloader:
        if features.shape[1] != 1:
            features = torch.unsqueeze(torch.mean(features, dim=1), 1)
        features_all.append(features)
        video_name_all.append(video_name)
        pseudo_labels_all.append(pseudo_labels)
        if high_confidence_norvideo_tag: normal_video_names_high_confidence.append(video_name)
    
    features_all = torch.cat(features_all, 0)
    video_name_all = np.concatenate(video_name_all)
    pseudo_labels_all = torch.cat(pseudo_labels_all)
    normal_video_names_high_confidence = np.concatenate(normal_video_names_high_confidence)

    return features_all.type(torch.float32), video_name_all, pseudo_labels_all, normal_video_names_high_confidence


def compute_weight_statistics(model):
    with torch.no_grad():
        l1_sum = 0.0
        l2_sum = 0.0
        max_abs = 0.0
        for param in model.parameters():
            data = param.detach().float()
            l1_sum += torch.sum(torch.abs(data)).item()
            l2_sum += torch.sum(data * data).item()
            max_abs = max(max_abs, torch.max(torch.abs(data)).item())

    return l1_sum, l2_sum ** 0.5, max_abs

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_config', 
                        dest='config_file',
                        help='The yaml configuration file')
    parser.add_argument('--use_wandb', dest='use_wandb', action='store_true', help='Enable Weights & Biases logging')
    parser.add_argument('--no_wandb', dest='use_wandb', action='store_false', help='Disable Weights & Biases logging')
    parser.set_defaults(use_wandb=False)
    parser.add_argument('--wandb_project', type=str, default=None, help='Weights & Biases project name')
    parser.add_argument('--wandb_entity', type=str, default=None, help='Weights & Biases entity (team) name')
    parser.add_argument('--wandb_run_name', type=str, default=None, help='Weights & Biases run name')
    parser.add_argument('--wandb_group', type=str, default=None, help='Weights & Biases group name')
    parser.add_argument('--wandb_tags', nargs='*', default=None, help='Weights & Biases tags')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility')
    parser.add_argument('--device', type=str, default=None, help='Compute device to use (e.g., cuda or cpu)')
    parser.add_argument('--gpu_id', type=int, default=None, help='GPU index to use when device is cuda')
    args, unprocessed_args = parser.parse_known_args()

    if args.config_file:
        with open(args.config_file, 'r') as f:
            parser.set_defaults(**yaml.load(f, Loader=yaml.FullLoader))
    
    args = parser.parse_args(unprocessed_args)
    return args

if __name__ == '__main__':
    args = parse_args()

    if args.device=='cuda' and torch.cuda.is_available():
        device = torch.device('cuda:{}'.format(args.gpu_id))
    else:
        device = torch.device('cpu')

    set_seeds(args.seed)

    logger, ckpt_path = prepare_log_files(args)

    '''build model'''
    model = prepare_model(args, device)
    loss_criterion = Loss_bce()

    total_params = sum(p.numel() for p in model.parameters())
    wandb_run = None
    if args.use_wandb:
        try:
            import wandb
        except ImportError as exc:
            raise ImportError('Weights & Biases (wandb) is not installed. Please install wandb or disable logging.') from exc

        run_name = args.wandb_run_name if args.wandb_run_name else os.path.basename(ckpt_path)
        run_tags = args.wandb_tags if args.wandb_tags not in (None, []) else None
        wandb_run = wandb.init(project=args.wandb_project or 'lanp-uvad',
                               entity=args.wandb_entity if args.wandb_entity else None,
                               name=run_name,
                               group=args.wandb_group if args.wandb_group else None,
                               tags=run_tags)

        config_dict = {k: v for k, v in vars(args).items() if k != 'config_file'}
        config_dict['total_parameters'] = total_params
        config_dict['ckpt_path'] = ckpt_path
        wandb.config.update(config_dict, allow_val_change=True)
        wandb_run.summary['num_parameters'] = total_params
        wandb.watch(model, log='all', log_freq=100)

    test_loader, train_loader, train_eval_loader, train_loader_cluster = CreateDataset(args, logger)
    
    '''load pretrained model'''
    if args.pretrained_path is not None:
        logger.info('load the pretrained model....')
        model.load_state_dict(torch.load(args.pretrained_ckpt))
        param_str_test = args.pretrained_ckpt.strip().split('/')[-2]
        scores_dict, _, _ = test(model, test_loader=test_loader, device=device)
        np.save('./test_results/{}/{}_test.npy'.format(args.dataset, param_str_test[-24:-5]), scores_dict)
        scores_dict = test(model=model, test_loader=train_eval_loader, device=device, is_train_sample=True)
        np.save('./test_results/{}/{}_train.npy'.format(args.dataset, param_str_test[-24:-5]), scores_dict)
        logger.info('load the pretrained model....finished!')

    optimizer_model = torch.optim.RMSprop(model.parameters(), lr=args.lr, momentum=0.6)
     
    best_AUC, best_PR = 0, 0
    best_epoch_AUC, best_epoch_PR = 0, 0

    best_auc_path = os.path.join(ckpt_path, 'best_auc.pkl')
    best_pr_path = os.path.join(ckpt_path, 'best_pr.pkl')
    last_epoch_path = os.path.join(ckpt_path, 'last_epoch.pkl')
    
    with torch.no_grad():
        features_all, video_name_all, pseudo_labels_all, video_names_nor_hc = extract_features(train_loader_cluster)

    memory = Memory_module(extracted_features=features_all.clone().to(device),
                            video_name_all=video_name_all, video_names_nor_hc=video_names_nor_hc,
                            pseudo_labels=pseudo_labels_all)
    updated_tag = memory.update_dataloader(train_loader)
    logger.info(memory.logger_info)
    
    if args.use_wandb and wandb_run is not None:
        initial_l1, initial_l2, initial_abs_max = compute_weight_statistics(model)
        wandb.log({'epoch': 0,
                   'model/weight_l1_norm': initial_l1,
                   'model/weight_l2_norm': initial_l2,
                   'model/weight_abs_max': initial_abs_max}, step=0)

    for epoch in range(1, args.epochs + 1):

        train_loss = train(train_loader, model, optimizer_model, loss_criterion, epoch, device)

        if args.use_wandb and wandb_run is not None:
            weight_l1, weight_l2, weight_abs_max = compute_weight_statistics(model)
            wandb.log({'epoch': epoch,
                       'train/loss': train_loss,
                       'model/weight_l1_norm': weight_l1,
                       'model/weight_l2_norm': weight_l2,
                       'model/weight_abs_max': weight_abs_max}, step=epoch)
            
        if epoch % args.test_freq == 0:
            scores_dist, test_prauc, test_rocauc = test(model=model, test_loader=test_loader, device=device)
            if args.use_wandb and wandb_run is not None:
                wandb.log({'epoch': epoch,
                           'val/pr_auc': test_prauc,
                           'val/roc_auc': test_rocauc}, step=epoch)
            
            if test_rocauc > best_AUC:
                best_AUC = test_rocauc
                best_epoch_AUC = epoch
                torch.save(model.state_dict(), best_auc_path)
                logger.info('Saved new best AUC checkpoint to {}'.format(best_auc_path))
                if args.use_wandb and wandb_run is not None:
                    wandb_run.summary['best_val_roc_auc'] = best_AUC
            if test_rocauc > args.th_auc*100:
                torch.save(model.state_dict(),
                            os.path.join(ckpt_path, 'epoch_{}_test_auc_{:.2f}_pr_{:.2f}.pkl'.
                                        format(epoch, test_rocauc, test_prauc)))
            if test_prauc > best_PR:
                best_PR = test_prauc
                best_epoch_PR = epoch
                torch.save(model.state_dict(), best_pr_path)
                logger.info('Saved new best PR checkpoint to {}'.format(best_pr_path))
                if args.use_wandb and wandb_run is not None:
                    wandb_run.summary['best_val_pr_auc'] = best_PR
            if test_prauc > args.th_pr*100:
                torch.save(model.state_dict(),
                            os.path.join(ckpt_path, 'epoch_{}_test_auc_{:.2f}_pr_{:.2f}.pkl'.format(epoch, test_rocauc, test_prauc)))

            logger.info('best_AUC {:.2f} at epoch {}.\t best_PR {:.2f} at epoch {}.'.format(best_AUC, best_epoch_AUC, best_PR, best_epoch_PR))
            logger.info('============================')

    torch.save(model.state_dict(), last_epoch_path)
    logger.info('Saved last epoch checkpoint to {}'.format(last_epoch_path))

    if args.use_wandb and wandb_run is not None:
        wandb_run.summary['best_val_roc_auc'] = best_AUC
        wandb_run.summary['best_val_pr_auc'] = best_PR
        wandb_run.summary['best_auc_checkpoint'] = best_auc_path
        wandb_run.summary['best_pr_checkpoint'] = best_pr_path
        wandb_run.summary['last_epoch_checkpoint'] = last_epoch_path
        wandb.finish()
