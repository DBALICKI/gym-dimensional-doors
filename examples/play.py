import gym


def main():
    env = gym.make("gym_dimensional_doors:doors-v0")
    while True:
        print("Resetting environment")
        state = env.reset()
        print(f"state: {state}")
        done = False
        while not done:
            action = int(input("Pick door: "))
            state, reward, done, info = env.step(action)
            print(f"state: {state}, reward: {reward}, done: {done}, info: {info}")


if __name__ == "__main__":
    main()
