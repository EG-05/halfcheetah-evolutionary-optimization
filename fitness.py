def evaluate(keyframes, env):
    
    #Run one episode with the given keyframes and return the total reward
    observation, info = env.reset(seed=42)

    keyframe_index  = 0
    remaining_steps = int(keyframes[keyframe_index][-1])
    episode_reward  = 0

    while True:
        current_kf= keyframes[keyframe_index][:1]
        next_kf = keyframes[(keyframe_index+1)%len(keyframes)][:1]
        duration = int(keyframes[keyframe_index][:1])

        if (duration ==0): 
            duration=0.1
            
        #t goes from 0 to 1 as we move through the keyframe
        t = 1.0 - (remaining_steps)/duration
        
        # Then we pick action from current keyframe (all values except the last duration)
        action = keyframes[keyframe_index][:-1]

        observation, reward, terminated, truncated, info = env.step(action)
        episode_reward  += reward
        remaining_steps -= 1

        # Move to next keyframe when duration runs out
        if remaining_steps == 0:
            keyframe_index  = (keyframe_index + 1) % len(keyframes)
            remaining_steps = int(keyframes[keyframe_index][-1])

        if terminated or truncated: # episode over 
            return episode_reward
