import gym
from keras.models import load_model, model_from_json
import time

json_file = open('cartpole-v0.json', 'r')
loaded_model_json = json_file.read()
json_file.close()

model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights("cartpole-v0.h5")
model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])



env = gym.make('CartPole-v0')

for i in range(100):
    observation = env.reset()
    fitness = 0
    
    for j in range(1000):
        action = model.predict(observation.reshape(1,4))[0][0]
        
        if action > 0.5: 
            action = 1 
        else: 
            action = 0

        env.render()
        observation, reward, done, _ = env.step(action)
        time.sleep(0.01)
        if done:
            break

        fitness += 1
        
    print(fitness)
env.close()