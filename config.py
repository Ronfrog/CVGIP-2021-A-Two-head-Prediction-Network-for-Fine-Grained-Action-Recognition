import argparse

def get_args():
    parser = argparse.ArgumentParser("Basketball Fouls Detection System Configurations.")
    # file path.
    parser.add_argument("--train_data_root", default="/home/chou/ronfrog/basketball_fouls/clips/", type=str)
    parser.add_argument("--train_data_json", default="/home/chou/ronfrog/basketball_fouls/clips/train1.json", type=str)
    parser.add_argument("--eval_data_root", default="/home/chou/ronfrog/basketball_fouls/clips/", type=str)
    parser.add_argument("--eval_data_json", default="/home/chou/ronfrog/basketball_fouls/clips/eval1.json", type=str)

    parser.add_argument("--data_size", default=(64, 3, 224, 224), type=tuple,
        help="[T, C, H, W]")
    parser.add_argument("--select_type", default="frequency", type=str, 
        choices=["frequency", "equally", "random"])
    parser.add_argument("--select_freq", default=1, type=int)
    parser.add_argument("--select_random_start", default=False, type=bool)
    parser.add_argument("--select_random_interval", default=1, type=int)

    # training size
    parser.add_argument("--batch_size", default=2, type=int)
    parser.add_argument("--num_workers", default=2, type=int)
    parser.add_argument("--backward_freq", default=8, type=float)
    
    # model
    parser.add_argument("--model_name", default="ours-resnet", type=str,
        choices=["ours", "ours-resnet", "ours-i3d", "resnet", "resnet2+1d", "i3d", "wide_resnet"])
    parser.add_argument("--num_scores", default=8, type=int)
    parser.add_argument("--num_classes", default=2, type=int)

    # learning rate and epoch. # log
    parser.add_argument("--base_lr", default=0.001, type=float)
    parser.add_argument("--warmup_epochs", default=10, type=int)
    parser.add_argument("--max_epochs", default=160, type=int, help="max training epoch")
    parser.add_argument("--eval_freq", default=5, type=int, help="evaluate model frequency(epochs).")
    parser.add_argument("--log_freq", default=50, type=int, help="show and save log frequency(batchs).")

    # loss function
    parser.add_argument("--use_info_gain", default=False, type=bool)
    parser.add_argument("--use_focal_loss", default=False, type=bool)

    # save model
    parser.add_argument("--save_model_root", default="./records/backup/", type=str)
    parser.add_argument("--trainlog_root", default="./records/train.log", type=str)
    parser.add_argument("--linearlog_root", default="./records/linear.log", type=str)
    parser.add_argument("--evallog_root", default="./records/eval.log", type=str)
    
    # load pretrained
    parser.add_argument("--load_imagenet_pretrained", default=False, type=bool, help="backbone resnet pretrained.")
    parser.add_argument("--load_pretrained", default=False, type=bool, help="load pretrained.")
    parser.add_argument("--pt_path", default="", type=str, help="pretrained .pt file path.")

    args = parser.parse_args()

    return args

