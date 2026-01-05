class LogicStage:
    """
    The Core Logic Layer (Logic Stagnation Zone).
    Aggregates all actors and manages the simulation state.
    """
    def __init__(self):
        self.actors = []

    def add_actor(self, actor):
        self.actors.append(actor)
        print(f"Registered actor: {actor.name}")

    def update(self, dt):
        """
        Update all actors in the stage.
        """
        for actor in self.actors:
            actor.update(dt)

    def get_actors(self):
        return self.actors
