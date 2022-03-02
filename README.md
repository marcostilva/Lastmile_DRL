# Lastmile_DRL
Deep reinforcement learning approach to Last-mile delivery with crowdshipping

We study a setting in which a company not only has a fleet of capacitated vehicles and drivers available to make 
deliveries but may also use the services of 
occasional drivers (ODs) who are willing to make deliveries using their vehicles in return for a small compensation. 
Under such a business model, a.k.a crowd shipping, the company seeks to make all the deliveries at 
the minimum total cost, i.e., the cost associated with their vehicles and drivers plus the compensation paid to the ODs.


Crowd shipping gives rise to new variants of the routing problem. It has been addressed in the literature as an extension of the classical vehicle routing problem or the traveling salesman problem, being modeled under different deterministic, stochastic, and/or dynamic optimization approaches. 
These approaches, in general, lead to reformulations that are hard to solve.

We consider a stochastic and dynamic last-mile delivery environment in which delivery orders, as well as ODs 
willing to make deliveries, arrive randomly throughout the day and present themselves for deliveries made within fixed time windows. 

We present a novel deep reinforcement learning (DRL) approach to the problem that can deal with large 
real-life problem instances, with a concern on the quality of the solution provided. To provide good solutions, a 
challenge associated with the design of the DRL approach concerns the action space, due to 
its combinatorial nature. Reinforcement learning algorithms typically require an action space that is small enough to enumerate or 
is continuous. We present a different approach where we formulate the action selection problem  as 
a mixed-integer optimization program. We combine the combinatorial structure of the action space with the neural architecture of 
the learned value function, involving techniques from machine learning and integer optimization.

Here we present the base code used to run DRL algorithms.
