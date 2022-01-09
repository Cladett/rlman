from gym.envs.registration import registry, register, make, spec

register(
		id='dVRLReach-v0',
		entry_point='dVRL_simulator.environments:PSMReachEnv',
		max_episode_steps=100,
)


register(
		id='dVRLPick-v0',
		entry_point='dVRL_simulator.environments:PSMPickEnv',
		max_episode_steps=100,
	)


register(
		id='dVRLPickPlace-v0',
		entry_point='dVRL_simulator.environments:PSMPickPlaceEnv',
		max_episode_steps=200,
	)


register(
		id='dVRLPickRail-v0',
		entry_point='dVRL_simulator.environments:PSMPickRailEnv',
		max_episode_steps=100,
	)

register(
		id='dVRLReachRail-v0',
		entry_point='dVRL_simulator.environments:PSMReachRailEnv',
		max_episode_steps=100,
	)

register(
		id='dVRLPickPlaceTarget-v0',
		entry_point='dVRL_simulator.environments:PSMPickPlaceTargetEnv',
		max_episode_steps=100,
	)

register(
		id='dVRLPickPlaceTargetObs-v0',
		entry_point='dVRL_simulator.environments:PSMPickPlaceTargetObsEnv',
		max_episode_steps=100,
	)

register(
		id='dVRLPickPlaceTargetEval-v0',
		entry_point='dVRL_simulator.environments:PSMPickPlaceTargetEvalEnv',
		max_episode_steps=100,
	)

register(
                id='dVRLReachImage-v0',
                entry_point='dVRL_simulator.environments:PSMReachImageEnv',
                max_episode_steps=30,
)

register(
		id='dVRLReachKidney-v0',
		entry_point='dVRL_simulator.environments:PSMReachKidneyEnv',
		max_episode_steps=100,
	)


############ Creating envs for running experiments in parallel ################
register(
		id='dVRLPickPlaceTargetE1-v0',
		entry_point='dVRL_simulator.environments:PSMPickPlaceTargetE1Env',
		max_episode_steps=100,
	)
register(
		id='dVRLPickPlaceTargetE2-v0',
		entry_point='dVRL_simulator.environments:PSMPickPlaceTargetE2Env',
		max_episode_steps=100,
	)
register(
		id='dVRLPickPlaceTargetE3-v0',
		entry_point='dVRL_simulator.environments:PSMPickPlaceTargetE3Env',
		max_episode_steps=100,
	)
register(
		id='dVRLPickPlaceTargetE4-v0',
		entry_point='dVRL_simulator.environments:PSMPickPlaceTargetE4Env',
		max_episode_steps=100,
	)
register(
		id='dVRLPickPlaceTargetE5-v0',
		entry_point='dVRL_simulator.environments:PSMPickPlaceTargetE5Env',
		max_episode_steps=100,
	)
register(
		id='dVRLPickPlaceTargetE6-v0',
		entry_point='dVRL_simulator.environments:PSMPickPlaceTargetE6Env',
		max_episode_steps=100,
	)
register(
		id='dVRLPickPlaceTargetEvalE1-v0',
		entry_point='dVRL_simulator.environments:PSMPickPlaceTargetEvalE1Env',
		max_episode_steps=100,
	)
register(
		id='dVRLPickPlaceTargetEvalE2-v0',
		entry_point='dVRL_simulator.environments:PSMPickPlaceTargetEvalE2Env',
		max_episode_steps=100,
	)
register(
		id='dVRLPickPlaceTargetEvalE3-v0',
		entry_point='dVRL_simulator.environments:PSMPickPlaceTargetEvalE3Env',
		max_episode_steps=100,
	)
register(
		id='dVRLPickPlaceTargetEvalE4-v0',
		entry_point='dVRL_simulator.environments:PSMPickPlaceTargetEvalE4Env',
		max_episode_steps=100,
	)
register(
		id='dVRLPickPlaceTargetEvalE5-v0',
		entry_point='dVRL_simulator.environments:PSMPickPlaceTargetEvalE5Env',
		max_episode_steps=100,
	)
register(
		id='dVRLPickPlaceTargetEvalE6-v0',
		entry_point='dVRL_simulator.environments:PSMPickPlaceTargetEvalE6Env',
		max_episode_steps=100,
	)
