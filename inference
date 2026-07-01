import os
import sys

# ==================== 默认参数注入 ====================
# 当直接右键运行，没有命令行参数时自动补充
if len(sys.argv) == 1:
    sys.argv += [
        '--cfg', 'config/BLSTF/BLSTF_PEMS08.py',   # 配置文件路径
        '--ckpt', 'checkpoints -yuan/BLSTF08_60/c92c93078dab4e87df4a139dec7c9288/BLSTF08_best_val_MAE.pt',  # 权重路径
        '--gpus', '0'  # 使用 GPU 0
    ]

# 打印工作目录，方便调试路径
print("Current working directory:", os.getcwd())

# ==================== 添加项目路径 ====================
sys.path.append(os.path.abspath(os.path.join(__file__, "../..")))

from argparse import ArgumentParser
from easytorch import launch_runner, Runner

# ==================== 解析参数 ====================
def parse_args():
    parser = ArgumentParser(description='EasyTorch Inference')
    parser.add_argument('-c', '--cfg', help='training config')
    parser.add_argument('--ckpt', help='checkpoint path', type=str)
    parser.add_argument('--gpus', default='0', help='visible gpus')
    return parser.parse_args()

# ==================== 主函数 ====================
def main(cfg: dict, runner: Runner, ckpt: str = None):
    runner.init_logger(logger_name='easytorch-inference', log_file_name='validate_result')
    runner.load_model(ckpt_path=ckpt)
    runner.test_process(cfg)

# ==================== 脚本入口 ====================
if __name__ == '__main__':
    args = parse_args()
    try:
        # 新版 easytorch
        launch_runner(args.cfg, main, (args.ckpt,), devices=args.gpus)
    except TypeError as e:
        # 兼容早期版本 easytorch
        if "launch_runner() got an unexpected keyword argument" in repr(e):
            launch_runner(args.cfg, main, (args.ckpt,), gpus=args.gpus)
        else:
            raise e
