import pygame
import random
import numpy as np
import os
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

FPS = 60
SCREENWIDTH = 288.0
SCREENHEIGHT = 512.0

# image and hitmask dicts
IMAGES, HITMASKS = {}, {}

load_saved_pool = 1
save_current_pool = 1
current_pool = []
fitness = []
total_models = 10

generation = 1

# Create directory for saving models if it doesn't exist
os.makedirs("Current_Model_Pool", exist_ok=True)

class MLXModel:
    def __init__(self, input_dim=3, hidden_dim=7, output_dim=2):
        # Define the model architecture
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Sigmoid(),
            nn.Linear(hidden_dim, output_dim),
            nn.Softmax()
        )
        
        # Initialize optimizer
        self.optimizer = optim.SGD(learning_rate=0.01, momentum=0.9)
        
    def get_weights(self):
        # Return a list of weights as numpy arrays for genetic operations
        return [p for p in self.model.parameters()]
        
    def set_weights(self, weights):
        # Set weights from numpy arrays
        params = list(self.model.parameters())
        for i, w in enumerate(weights):
            params[i][:] = mx.array(w)
    
    def save_weights(self, filename):
        # Save weights to file
        weights = self.get_weights()
        np.save(filename, weights)
        
    def load_weights(self, filename):
        # Load weights from file
        if os.path.exists(filename):
            weights = np.load(filename, allow_pickle=True)
            self.set_weights(weights)
            return True
        return False
        
    def predict(self, x, verbose=0):
        # Convert input to MLX array
        x_mlx = mx.array(x, dtype=mx.float32)
        
        # Forward pass
        output = self.model(x_mlx)
        
        # Convert back to numpy for compatibility
        return [output]

    def save_pool():
        for xi in range(total_models):
            current_pool[xi].save_weights(f"Current_Model_Pool/first_model{xi}.npy")
        print("Saved current pool!")

    def model_crossover(model_idx1, model_idx2):
        global current_pool
        weights1 = current_pool[model_idx1].get_weights()
        weights2 = current_pool[model_idx2].get_weights()
        weightsnew1 = [np.copy(w) for w in weights1]
        weightsnew2 = [np.copy(w) for w in weights2]
        
        # Simple crossover: swap the first layer weights
        weightsnew1[0] = weights2[0]
        weightsnew2[0] = weights1[0]
        
        return np.array([weightsnew1, weightsnew2], dtype=object)

def model_mutate(weights):
    """
    Safely mutate network weights with proper type checking
    """
    # Create a copy to avoid modifying the original weights
    weights_copy = [np.copy(w) for w in weights]
    
    for xi in range(len(weights_copy)):
        # Skip if weight is not a numeric array
        if not isinstance(weights_copy[xi], np.ndarray) or weights_copy[xi].dtype.kind not in 'iufc':
            continue
            
        # For scalars (0-d arrays)
        if weights_copy[xi].ndim == 0:
            if random.uniform(0, 1) > 0.85:
                change = random.uniform(-0.5, 0.5)
                weights_copy[xi] = weights_copy[xi] + change
        # For arrays with dimensions
        elif weights_copy[xi].ndim >= 1:
            # Create flattened view of array for iteration
            flat_view = weights_copy[xi].flat
            for i in range(len(flat_view)):
                if random.uniform(0, 1) > 0.85:
                    change = random.uniform(-0.5, 0.5)
                    flat_view[i] = flat_view[i] + change
                    
    return weights_copy

def model_crossover(model_idx1, model_idx2):
    """
    Perform crossover between two parent models with enhanced error checking
    """
    global current_pool
    weights1 = current_pool[model_idx1].get_weights()
    weights2 = current_pool[model_idx2].get_weights()
    
    # Ensure we're only copying numeric arrays
    weightsnew1 = []
    weightsnew2 = []
    
    for w in weights1:
        if isinstance(w, np.ndarray) and w.dtype.kind in 'iufc':
            weightsnew1.append(np.copy(w))
        else:
            weightsnew1.append(w)
            
    for w in weights2:
        if isinstance(w, np.ndarray) and w.dtype.kind in 'iufc':
            weightsnew2.append(np.copy(w))
        else:
            weightsnew2.append(w)
    
    # Perform crossover only if the arrays are compatible
    if len(weightsnew1) > 0 and len(weightsnew2) > 0:
        # Simple crossover: swap the first layer weights if possible
        if (isinstance(weightsnew1[0], np.ndarray) and 
            isinstance(weightsnew2[0], np.ndarray) and
            weightsnew1[0].shape == weightsnew2[0].shape):
            temp = weightsnew1[0].copy()
            weightsnew1[0] = weightsnew2[0].copy()
            weightsnew2[0] = temp
    
    return np.array([weightsnew1, weightsnew2], dtype=object)

def save_pool():
    """
    Save the current model pool with error handling
    """
    try:
        for xi in range(total_models):
            current_pool[xi].save_weights(f"Current_Model_Pool/first_model{xi}.npy")
        print("Saved current pool!")
    except Exception as e:
        print(f"Error saving pool: {e}")

def showGameOverScreen():
    """Perform genetic updates here"""
    global current_pool
    global fitness
    global generation
    
    # Display a message on screen while genetic algorithms runs
    font = pygame.font.SysFont(None, 36)
    text = font.render("Evolving new generation...", True, (255, 255, 255))
    text_rect = text.get_rect(center=(SCREENWIDTH/2, SCREENHEIGHT/2))
    SCREEN.blit(IMAGES['background'], (0, 0))
    SCREEN.blit(text, text_rect)
    pygame.display.update()
    
    new_weights = []
    total_fitness = sum(fitness)
    
    if total_fitness == 0:  # Prevent division by zero
        for i in range(total_models):
            fitness[i] = 1
        total_fitness = total_models
    
    print("Generation:", generation)
    print("Total fitness:", total_fitness)
    
    # Convert fitness to probability
    normalized_fitness = [fit / total_fitness for fit in fitness]
    cumulative_fitness = [sum(normalized_fitness[:i+1]) for i in range(total_models)]
    
    try:
        for select in range(int(total_models/2)):
            parent1 = random.uniform(0, 1)
            parent2 = random.uniform(0, 1)
            
            # Find parents using roulette wheel selection
            idx1 = next((i for i, val in enumerate(cumulative_fitness) if val >= parent1), total_models-1)
            idx2 = next((i for i, val in enumerate(cumulative_fitness) if val >= parent2), total_models-1)
            
            # Crossover
            new_weights1 = model_crossover(idx1, idx2)
            
            # Mutate with error handling
            try:
                updated_weights1 = model_mutate(new_weights1[0])
                updated_weights2 = model_mutate(new_weights1[1])
                
                new_weights.append(updated_weights1)
                new_weights.append(updated_weights2)
            except Exception as e:
                print(f"Error during mutation: {e}")
                # Fall back to using original weights if mutation fails
                new_weights.append(new_weights1[0])
                new_weights.append(new_weights1[1])
    
        # Apply new weights to models
        for select in range(min(len(new_weights), total_models)):
            fitness[select] = -100
            try:
                current_pool[select].set_weights(new_weights[select])
            except Exception as e:
                print(f"Error setting weights for model {select}: {e}")
                # Create a fresh model if setting weights fails
                current_pool[select] = MLXModel(input_dim=3, hidden_dim=7, output_dim=2)
        
        # Save the current model pool
        if save_current_pool == 1:
            save_pool()
        
        generation += 1
    except Exception as e:
        print(f"Error in genetic algorithm: {e}")
        # Ensure generation still increments even if there's an error
        generation += 1
    
    return
def predict_action(agentX, rocketX, rocketY, model_num):
    global current_pool
    # Normalize inputs
    agentX = min(SCREENWIDTH, agentX) / SCREENWIDTH - 0.5
    rocketX = min(SCREENWIDTH, rocketX) / SCREENWIDTH - 0.5
    rocketY = min(SCREENHEIGHT, rocketY) / SCREENHEIGHT - 0.5
    
    neural_input = np.asarray([agentX, rocketX, rocketY])
    neural_input = np.atleast_2d(neural_input)
    
    # Use the model to predict the action
    output_prob = current_pool[model_num].predict(neural_input, verbose=0)[0]
    
    return np.argmax(output_prob)

# Initialize all models
def initialize_models():
    global current_pool, fitness, total_models
    
    current_pool = []
    fitness = []
    
    for i in range(total_models):
        model = MLXModel(input_dim=3, hidden_dim=7, output_dim=2)
        current_pool.append(model)
        fitness.append(-100)

    if load_saved_pool:
        try:
            success = False
            for i in range(total_models):
                if current_pool[i].load_weights(f"Current_Model_Pool/first_model{i}.npy"):
                    success = True
            if success:
                print("Loaded saved pool successfully!")
            else:
                print("Could not load saved pool. Starting with new models.")
        except Exception as e:
            print(f"Error loading saved pool: {e}. Starting with new models.")

def main():
    global SCREEN, FPSCLOCK
    
    # Initialize models
    initialize_models()
    
    pygame.init()
    FPSCLOCK = pygame.time.Clock()
    SCREEN = pygame.display.set_mode((int(SCREENWIDTH), int(SCREENHEIGHT)))
    pygame.display.set_caption('Evolutionary Spaceship MLX')

    # Load images
    try:
        IMAGES['background'] = pygame.image.load('img/background.png').convert()
        IMAGES['agent'] = pygame.image.load('img/good_spaceship.png').convert_alpha()
        IMAGES['rocket'] = pygame.image.load('img/rocket.png').convert_alpha()
    except pygame.error as e:
        print(f"Couldn't load image: {e}")
        # Create placeholder images if originals not found
        IMAGES['background'] = pygame.Surface((int(SCREENWIDTH), int(SCREENHEIGHT)))
        IMAGES['background'].fill((0, 0, 0))
        IMAGES['agent'] = pygame.Surface((20, 20), pygame.SRCALPHA)
        pygame.draw.rect(IMAGES['agent'], (0, 255, 0), (0, 0, 20, 20))
        IMAGES['rocket'] = pygame.Surface((15, 25), pygame.SRCALPHA)
        pygame.draw.rect(IMAGES['rocket'], (255, 0, 0), (0, 0, 15, 25))

    # Get hitmasks
    HITMASKS['agent'] = getHitmask(IMAGES['agent'])
    HITMASKS['rocket'] = getHitmask(IMAGES['rocket'])
    
    while True:
        gameloop()
        showGameOverScreen()

def gameloop():
    global fitness
    score = 0
    
    # Players 
    playersState = [] # True (alive) or False (dead)
    playersPosition = []
    playersVelX = []
    spaceship_acceleration = 8 
    
    for idx in range(total_models):
        playersPosition.append([(SCREENWIDTH * 0.45), (SCREENHEIGHT * 0.80)])
        playersVelX.append(0)
        playersState.append(True)

    # Rocket
    rocket = getRandomRocket()
    rocketVelY = 8 

    alive_players = total_models
    
    while True:                              
        for event in pygame.event.get(): # QUIT THE GAME
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            # Keyboard 
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    playersVelX[0] = -spaceship_acceleration
                if event.key == pygame.K_RIGHT:
                    playersVelX[0] = spaceship_acceleration
            if event.type == pygame.KEYUP:
                if event.key == pygame.K_LEFT or event.key == pygame.K_RIGHT:
                    playersVelX[0] = 0

        # check for crash here, return status list
        crashTest = checkCrash(playersPosition, rocket)

        for idx in range(total_models):
            if playersState[idx] == True and crashTest[idx] == True:
                alive_players -= 1
                playersState[idx] = False
            if playersState[idx] == True:
                fitness[idx] += 1
                action = predict_action(playersPosition[idx][0], rocket[0], rocket[1], idx)
                if action == 0:
                    playersVelX[idx] = -spaceship_acceleration
                elif action == 1:
                    playersVelX[idx] = spaceship_acceleration
                
                nextPlayerPosition = (playersPosition[idx][0] + playersVelX[idx])
                if nextPlayerPosition + IMAGES['agent'].get_width() > SCREENWIDTH or nextPlayerPosition < 0: 
                    continue
                playersPosition[idx][0] += playersVelX[idx]
                    
        if alive_players == 0:
            return True
        
        # move rocket 
        rocket[1] += rocketVelY

        # check for scores
        for idx in range(total_models):
            if playersState[idx] == True and rocket[1] > SCREENHEIGHT:
                score += 1
                fitness[idx] += 25
                
        # get new rocket when it leaves the screen 
        if rocket[1] > SCREENHEIGHT:
            rocket = getRandomRocket()
            
        # draw background
        SCREEN.blit(IMAGES['background'], (0,0))

        # draw rocket
        SCREEN.blit(IMAGES['rocket'], (rocket[0], rocket[1]))

        # print score so player overlaps the score
        showScore(score)
        
        # draw player(s) 
        for idx, player in enumerate(playersPosition):
            if playersState[idx] == True:
                SCREEN.blit(IMAGES['agent'], player)
        
        # Show generation number
        showGeneration(generation)
        
        pygame.display.update()
        FPSCLOCK.tick(FPS)

def getRandomRocket():
    """returns a randomly generated rocket"""
    rocketX = random.randrange(0, int(SCREENWIDTH - IMAGES['rocket'].get_width()))
    rocketY = -IMAGES['rocket'].get_height()
    return [rocketX, rocketY]

def showScore(score):
    font = pygame.font.SysFont(None, 25)
    text = font.render("Dodged: " + str(score), True, (255, 255, 255))
    SCREEN.blit(text, (0, 0))

def showGeneration(gen):
    font = pygame.font.SysFont(None, 25)
    text = font.render("Generation: " + str(gen), True, (255, 255, 255))
    SCREEN.blit(text, (0, 25))

def checkCrash(players, rocket):
    """ return True if player crash with rocket """
    statuses = []
    for idx in range(total_models):
        statuses.append(False)
        
    for idx in range(total_models):
        playerW = IMAGES['agent'].get_width()
        playerH = IMAGES['agent'].get_height()
        playerRect = pygame.Rect(players[idx][0], players[idx][1],
                        playerW, playerH)
        rocketW = IMAGES['rocket'].get_width()
        rocketH = IMAGES['rocket'].get_height()
        rocketRect = pygame.Rect(rocket[0], rocket[1], rocketW, rocketH)

        rHitMask = HITMASKS['rocket']
        pHitMask = HITMASKS['agent']

        # agent <--> rocket collision
        rCollide = pixelCollision(playerRect, rocketRect, pHitMask, rHitMask)

        if rCollide:
            statuses[idx] = True
            
    return statuses

def showGameOverScreen():
    """Perform genetic updates here"""
    global current_pool
    global fitness
    global generation
    
    # Display a message on screen while genetic algorithms runs
    font = pygame.font.SysFont(None, 36)
    text = font.render("Evolving new generation...", True, (255, 255, 255))
    text_rect = text.get_rect(center=(SCREENWIDTH/2, SCREENHEIGHT/2))
    SCREEN.blit(IMAGES['background'], (0, 0))
    SCREEN.blit(text, text_rect)
    pygame.display.update()
    
    new_weights = []
    total_fitness = sum(fitness)
    
    if total_fitness == 0:  # Prevent division by zero
        for i in range(total_models):
            fitness[i] = 1
        total_fitness = total_models
    
    print("Generation:", generation)
    print("Total fitness:", total_fitness)
    
    # Convert fitness to probability
    normalized_fitness = [fit / total_fitness for fit in fitness]
    cumulative_fitness = [sum(normalized_fitness[:i+1]) for i in range(total_models)]
    
    for select in range(int(total_models/2)):
        parent1 = random.uniform(0, 1)
        parent2 = random.uniform(0, 1)
        
        # Find parents using roulette wheel selection
        idx1 = next((i for i, val in enumerate(cumulative_fitness) if val >= parent1), total_models-1)
        idx2 = next((i for i, val in enumerate(cumulative_fitness) if val >= parent2), total_models-1)
        
        # Crossover
        new_weights1 = model_crossover(idx1, idx2)
        
        # Mutate
        updated_weights1 = model_mutate(new_weights1[0])
        updated_weights2 = model_mutate(new_weights1[1])
        
        new_weights.append(updated_weights1)
        new_weights.append(updated_weights2)
    
    # Apply new weights to models
    for select in range(len(new_weights)):
        fitness[select] = -100
        current_pool[select].set_weights(new_weights[select])
    
    # Save the current model pool
    if save_current_pool == 1:
        save_pool()
    
    generation += 1
    return

def pixelCollision(rect1, rect2, hitmask1, hitmask2):
    """Checks if two objects collide and not just their rects"""
    rect = rect1.clip(rect2)

    if rect.width == 0 or rect.height == 0:
        return False

    x1, y1 = rect.x - rect1.x, rect.y - rect1.y
    x2, y2 = rect.x - rect2.x, rect.y - rect2.y

    for x in range(rect.width):
        for y in range(rect.height):
            if x1+x < len(hitmask1) and y1+y < len(hitmask1[0]) and \
               x2+x < len(hitmask2) and y2+y < len(hitmask2[0]):
                if hitmask1[x1+x][y1+y] and hitmask2[x2+x][y2+y]:
                    return True
    return False

def getHitmask(image):
    """returns a hitmask using an image's alpha."""
    mask = []
    for x in range(image.get_width()):
        mask.append([])
        for y in range(image.get_height()):
            mask[x].append(bool(image.get_at((x,y))[3]))
    return mask

if __name__ == '__main__':
    main()
