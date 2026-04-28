import os  # 读写环境变量，用于设置运行时兼容开关

import hydra  # Hydra 配置入口装饰器


@hydra.main(
    config_path="config",  # 指向小写 wmrl 根下的配置目录
    config_name="wmrl_qwenpi",  # 默认主配置文件名
    version_base=None,  # 不启用 Hydra 版本兼容层
)
def main(config):
    """Hydra 入口：将解析后的配置交给训练启动函数。"""
    run_wmrl(config)  # 直接调用启动逻辑，便于测试时复用


def run_wmrl(config):
    """WMRL 训练启动逻辑（支持单机与可选 Ray）。"""
    os.environ["ENSURE_CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "")

    use_ray = bool(config.runtime.get("use_ray", False))  # 从配置读取是否启用 Ray
    if use_ray:  # 仅在需要分布式时导入 Ray，避免无依赖环境报错
        import ray

        if not ray.is_initialized():  # 避免重复初始化
            ray.init(
                runtime_env={
                    "env_vars": {
                        "TOKENIZERS_PARALLELISM": "true",  # 减少 tokenizer 警告
                        "NCCL_DEBUG": "WARN",  # 保留 NCCL 警告信息
                    }
                }
            )

    from wmrl.trainer import RayWMRLTrainer  # 延迟导入，确保路径和环境已经就绪

    trainer = RayWMRLTrainer(config=config)  # 构建训练器
    trainer.fit()  # 执行训练主循环


if __name__ == "__main__":
    main()  # 允许 python -m main_wmrl_qwenpi 直接启动
