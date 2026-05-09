__all__ = ["JEPOActorRolloutWorker", "JEPOLewmRewardWorker"]


def __getattr__(name: str):
    if name == "JEPOActorRolloutWorker":
        from .actor_rollout_worker import JEPOActorRolloutWorker

        return JEPOActorRolloutWorker
    if name == "JEPOLewmRewardWorker":
        from .lewm_reward_worker import JEPOLewmRewardWorker

        return JEPOLewmRewardWorker
    raise AttributeError(name)
