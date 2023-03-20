from gym.envs.registration import register


register(
    id="doors-v0",
    entry_point="gym_dimensional_doors.envs:DoorsEnv",
    kwargs={
        "num_doors": 2,
        "num_stages": 10,
    },
)
