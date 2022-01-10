import numpy as np


def compute_reward(achieved_goal, goal):
    
    distance_threshold = 0.0045
    reward_type = 'sparse'
    grasp_reward = True
    d= np.linalg.norm(achieved_goal - goal, axis=-1)

    if reward_type == 'sparse':
        #returns = (np.bool_((d < distance_threshold) and grasp_reward)).astype(np.float32) - 1
        #returns = all((np.bool_((d < distance_threshold) and grasp_reward)).astype(np.float32) - 1)
        returns = np.bool_(np.logical_and((d < distance_threshold), grasp_reward)).astype(np.float32) - 1
        return returns 
    else:
        return -100 * d



def main():

    goal = np.array([[ 0.5774975 ,  0.7110739 , -0.08394451],
           [ 0.5847931 ,  0.6942034 , -0.07486077],
           [ 0.5222702 ,  0.5185175 ,  0.004144  ],
           [ 0.5247927 ,  0.5328512 ,  0.0028923 ],
           [ 0.51815987,  0.8967638 , -0.00822039],
           [ 0.51846504,  0.4658079 ,  0.0537899 ],
           [ 0.6391525 ,  0.7445574 , -0.04548759],
           [-0.7149434 ,  0.6736517 , -0.06626371],
           [-0.7149434 ,  0.6736517 , -0.06626371]]) 

    achieved_goal = np.array([[ 0.5770254 ,  0.84778786, -0.05851002],
           [ 0.5847931 ,  0.6942034 , -0.07486077],
           [ 0.5702257 ,  0.6173849 , -0.08110972],
           [ 0.64219   ,  0.7629585 , -0.08306237],
           [ 0.58108807,  0.9713173 , -0.03239126],
           [ 0.51846504,  0.4658079 ,  0.0537899 ],
           [ 0.6226969 ,  0.78255177, -0.0606987 ],
           [ 0.514431  ,  0.4859686 ,  0.03651171],
           [ 0.6384754 ,  0.739913  , -0.04587145]]) 

    #ret = compute_reward(achieved_goal[1] , goal[1])
    ret = compute_reward(achieved_goal , goal)
    print('returns', ret)

if __name__=='__main__':
    main()

