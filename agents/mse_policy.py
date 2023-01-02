
class MsePolicy():
    
    def __init__(self, actor_network):
        self.actor_network = actor_network

    def act(self, time_step):
        print(time_step)
        return self.actor_network.forward(time_step['observations'])