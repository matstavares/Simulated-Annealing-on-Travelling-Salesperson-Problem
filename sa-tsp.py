from cmath import inf
import math
import sys
import numpy
import pygame
from matplotlib import pyplot as plt
from beautifultable import BeautifulTable

# test to see how many possible ways there is
# from itertools import permutations
# ways=permutations(path)

INSTANCE = "instance100.in"
INSTANCE = "instance51.in"

TITLE = "Simulated Annealing on Travelling Sales Person problem"

LINE_THICKNESS = 2
CITY_RADIUS = 5
FIRST_CITY = 1

T0_INITIAL_TEMPERATURE = 50
TN_FINAL_TEMPERATURE = 0.0001
N_COOLING_CYCLES = 2000
THERMAL_BALANCE = 1000
ALPHA_FACTOR = 0.99

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREY = (190, 190, 190)
RED = (255, 10, 10)



class Node:
    def __init__(self, id, x, y):
        self.id = id
        self.pos = (x, y)
        self.x = x
        self.y = y
        self.cost = 0
        self._from = 0
        self._to = 0

    def get_pos(self, width, height):
        return ((self.x * width) + 50, (self.y * height) + 50)

    def draw(self, WINDOW, w, h, label):
        pygame.draw.circle(WINDOW, RED, self.get_pos(w, h), CITY_RADIUS)
        
        itemLabel = label.render(f"{self.id}", True, BLACK)
        margin_x = round(itemLabel.get_width() / 2) 
        margin_y = round(itemLabel.get_height() / 2) + 15
        WINDOW.blit(itemLabel, (self.get_pos(w, h)[0] - margin_x, 
                                self.get_pos(w, h)[1] - margin_y))
        pygame.display.update()


class Draw:
    def __init__(self, WINDOW, width, height):
        self.WINDOW = WINDOW
        self.width = width
        self.height = height
        self.label = pygame.font.SysFont("Helvetica", 15)

    def draw(self, path, cycle, T, speedup):
        self.WINDOW.fill(WHITE)
        
        distance = self.label.render(f"Distance: {get_distance(path):.2f}", True, BLACK)
        max_cycle = self.label.render(f"Max cycle: {N_COOLING_CYCLES}", True, BLACK)
        cur_cycle = self.label.render(f"Cycle: {cycle}", True, BLACK)
        temp_initial = self.label.render(f"Initial: {T0_INITIAL_TEMPERATURE:.2f}°", True, BLACK)
        temp_current = self.label.render(f"Temp: {T:.2f}°", True, BLACK)
        temp_final = self.label.render(f"Final: {TN_FINAL_TEMPERATURE}°", True, BLACK)
        self.WINDOW.blit(distance, (10, 10))
        self.WINDOW.blit(max_cycle, (10, 30))
        self.WINDOW.blit(cur_cycle, (10, 50))
        self.WINDOW.blit(temp_initial, (10, 70))
        self.WINDOW.blit(temp_current, (10, 90))
        self.WINDOW.blit(temp_final, (10, 110))

        if not speedup:
            for i in range(len(path)):
                pygame.draw.line(self.WINDOW, GREY, 
                                    path[i-1].get_pos(self.width, self.height),
                                    path[i].get_pos(self.width, self.height), LINE_THICKNESS)
                pygame.display.update()
            for i in range(len(path)):
                path[i].draw(self.WINDOW, self.width, self.height, self.label)
                pygame.display.update()
        pygame.display.update()

def events():
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False
            sys.exit()
        pygame.display.update()

def len_cities(file):
    with open(file, 'r') as f:
        mytiles = f.readlines()
        mytiles = [i.strip().split() for i in mytiles]
        count = 0
        max_x = 0
        max_y = 0
        for row, tiles in enumerate(mytiles):
            for col, tile in enumerate(tiles):
                if col == 0:
                    try:
                        id = int(tile)
                        count += 1
                    except ValueError:
                        break
                elif col == 1:
                    x = int(tile)
                    max_x = max(max_x, x)
                elif col == 2:
                    y = int(tile)
                    max_y = max(max_y, y)

    return count, max_x, max_y
                         
def read_file(file):
    with open(file, 'r') as f:
        mytiles = f.readlines()
        mytiles = [i.strip().split() for i in mytiles]
        cities = {}

        for row, tiles in enumerate(mytiles):
            for col, tile in enumerate(tiles):
                if col == 0:
                    try:
                        id = int(tile)
                    except ValueError:
                        break
                elif col == 1:
                    x = int(tile)
                elif col == 2:
                    y = int(tile)
                    node = Node(id, x, y)
                    # if id > 1:
                    #     node._from = id-1
                    # else:
                    #     node._from = 51
                    # if id < 51:
                    #     node._to = id+1
                    # else:
                    #     node._to = 1
                    cities.update({id : node})

    return cities

def generate_path(cities):
    cidades = 1
    city_source = numpy.random.randint(1, len(cities)+1)
    cidadinha = cities.copy()
    while cidades < len(cities):
        while True:
            city_destination = numpy.random.randint(1, len(cities)+1)
            if city_source != city_destination:
                if cities[city_destination]._to == 0 and \
                   cities[city_destination]._from == 0:
                    cities[city_source]._to = city_destination
                    cities[city_destination]._from = cities[city_source].id
                    city_source = city_destination
                    cidadinha.pop(city_source)
                    if len(cidadinha) == 1:
                        begin = list(cidadinha.keys())[0]
                        cities[city_source]._to = cities[begin].id
                        cities[begin]._from = cities[city_source].id
                    cidades += 1
                    break

def make_path(cities):
    path = []
    _from = FIRST_CITY
    counter = 0

    while counter < len(cities):
        path.append(cities[_from])
        _from = cities[_from]._to

        counter += 1
    return path

def get_distance(path):
    distance = 0
    for i in range(len(path)):
        x_delta = path[i-1].x - path[i].x
        y_delta = path[i-1].y - path[i].y
        distance += (x_delta ** 2 + y_delta ** 2) ** 0.5
    return distance

def get_distance_node(pos1, pos2):
    x, y = pos1
    a, b = pos2
    x_delta = x - a
    y_delta = y - b
    distance = (x_delta ** 2 + y_delta ** 2) ** 0.5
    return distance


def id_in_path(path, id):
    for i in range(len(path)):
        if path[i].id == id:
            return i

def next_to(path, id):
    ident = id_in_path(path, id)
    closest = [[inf, inf],[inf, inf]]
    for i in range(len(path)):
        if path[i].id != id:
            if get_distance_node(path[i].pos, path[ident].pos) < closest[0][0]:
                closest[0][0] = get_distance_node(path[i].pos, path[ident].pos)
                closest[1][0] = i
            elif get_distance_node(path[i].pos, path[ident].pos) < closest[0][1]:
                closest[0][1] = get_distance_node(path[i].pos, path[ident].pos)
                closest[1][1] = i

    return closest


def swapping_neighboors(path, number):
    
    cidadinha = path.copy()
    city_s = city_d = 0

    for i in range(number):        
        while city_s == city_d:
            city_s = numpy.random.randint(0, len(cidadinha))
            city_d = numpy.random.randint(0, len(cidadinha))
    
        cidadinha[city_s-2]._to = cidadinha[city_d-1].id
        cidadinha[city_d-1]._from = cidadinha[city_s-2].id

        cidadinha[city_d-1]._to = cidadinha[city_s].id
        cidadinha[city_s]._from = cidadinha[city_d-1].id

        cidadinha[city_d-2]._to = cidadinha[city_s-1].id
        cidadinha[city_s-1]._from = cidadinha[city_d-2].id

        cidadinha[city_s-1]._to = cidadinha[city_d].id
        cidadinha[city_d]._from = cidadinha[city_s-1].id

        temp_node = cidadinha[city_s-1]
        cidadinha[city_s-1] = cidadinha[city_d-1]
        cidadinha[city_d-1] = temp_node

        city_s = city_d = 0

    return cidadinha

def swapping_neighboors2(path, number):
    
    cidadinha = path.copy()
    city_s = city_d = 0

    for i in range(number):        
        city_s = numpy.random.randint(0, len(cidadinha))

        cidadinha[city_s-2]._to = cidadinha[city_s].id
        cidadinha[city_s]._from = cidadinha[city_s-2].id
        cidadinha[city_s-1]._to = cidadinha[city_s-2].id
        cidadinha[city_s-2]._from = cidadinha[city_s-1].id
        cidadinha[city_s-1]._from = cidadinha[city_s-3].id
        cidadinha[city_s-3]._to = cidadinha[city_s-1].id

        temp_node = cidadinha[city_s-1]
        cidadinha[city_s-1] = cidadinha[city_s-2]
        cidadinha[city_s-2] = temp_node

    return cidadinha

def swapping_neighboors3(path, number):
    
    cidadinha = path.copy()
    city_s = city_d = 0

    for i in range(number):        
        city_s = numpy.random.randint(0, len(cidadinha))

        closest = next_to(cidadinha, cidadinha[city_s].id)
        cd_1 = closest[1][0]
        cd_2 = closest[1][1]
        if cd_1 != cidadinha[city_s]._from and cd_1 != cidadinha[city_s]._to:

            ident = id_in_path(cidadinha, cidadinha[cd_1]._to)
            cidadinha[cd_1-1]._to = cidadinha[ident].id
            cidadinha[ident]._from = cidadinha[cd_1-1].id

            cidadinha[city_s-1]._to = cidadinha[cd_1].id 
            cidadinha[cd_1]._from = cidadinha[city_s-1].id
            cidadinha[cd_1]._to = cidadinha[city_s].id
            cidadinha[city_s]._from = cidadinha[cd_1].id

            temp_node = cidadinha[cd_1]
            cidadinha.pop(cd_1)
            cidadinha.insert(city_s, temp_node)

        elif cd_2 != cidadinha[city_s]._from and cd_2 != cidadinha[city_s]._to:
            
            ident = id_in_path(cidadinha, cidadinha[cd_2]._to)
            cidadinha[cd_2-1]._to = cidadinha[ident].id
            cidadinha[ident]._from = cidadinha[cd_2-1].id

            ident = id_in_path(cidadinha, cidadinha[city_s]._to)
            cidadinha[city_s]._to = cidadinha[cd_2].id 
            cidadinha[cd_2]._from = cidadinha[city_s].id
            cidadinha[cd_2]._to = cidadinha[ident].id
            cidadinha[ident]._from = cidadinha[cd_2].id

            temp_node = cidadinha[cd_2]
            cidadinha.pop(cd_2)
            cidadinha.insert(ident, temp_node)

    return cidadinha

def boltzman_factor(delta, T):
    #aproximation to boltzmann constant
    k = 1 
    return math.exp((-delta)/(k*T))

def sa(setting, alpha, SAmax, T0, TN, N, cooling, s, draw):
    s_better = s
    T = T0
    cycle = 0
    lastBetterCycle = 0
    lastBetterTemp = 0
    thermal_balance = SAmax
    iterations = []
    distances = []
    temp = []
    accepted = 0

    while (cycle < N or T > TN):
        cycle += 1
        
        thermal_movement = 0
        temp.append(T)

        while (thermal_movement < thermal_balance):
            thermal_movement += 1

            # swaping pairs
            number_pairs = numpy.random.randint(1,6)
            if T < T0*0.20:
                s_line = swapping_neighboors(s, number_pairs)
            else:
                s_line = swapping_neighboors3(s, number_pairs)

            events()
            delta = get_distance(s_line) - get_distance(s)
            
            if (delta < 0):
                s = s_line
                accepted += 1
                if (get_distance(s_line) < get_distance(s_better)):
                    s_better = s_line
                    iterations.append(cycle)
                    distances.append(round(get_distance(s_line), 1))
                    lastBetterCycle = cycle
                    lastBetterTemp = T
                    draw.draw(s_line, cycle, T, True)
            else:
                x = numpy.random.uniform(0,1)
                
                if (x < boltzman_factor(delta, T)):
                    accepted += 1
                    s = s_line
                
        if setting == True:
            if ((accepted / thermal_balance) < 0.94):
                global T0_INITIAL_TEMPERATURE
                T0_INITIAL_TEMPERATURE *= 1.1
                return True
            else:
                return False
        # Cooling schedule 0
        elif cooling == 0:
            T = T0 - (cycle*(T0-TN)/N)

        # Cooling schedule 1
        elif cooling == 1:
            T = T0 *((TN/T0)**(cycle/N))

        # Cooling schedule 2
        elif cooling == 2:
            A = ((T0 - TN) * (N + 1)) / N
            T = A/(cycle+1) + (T0 - A)

        # Cooling schedule 3
        elif cooling == 3:
            A = (math.log(T0 - TN)) / math.log(N)
            T = T0 - (cycle ** A)
        
        # Cooling schedule 4
        elif cooling == 4:
            T = ((T0 - TN) / (1 + math.exp(0.3*(cycle - (N/2))))) + TN
       
        # Cooling schedule 5
        elif cooling == 5:
            T = (((1/2)*(T0 - TN)) * (1 + math.cos((cycle * math.pi)/ N ))) + TN

        # Cooling schedule 6
        elif cooling == 6:
            T = (((1/2)*(T0 - TN)) * (1 - math.tanh(((10*cycle)/ N ) - 5))) + TN

        # Cooling schedule 7
        elif cooling == 7:
            T = ((T0 - TN) / (math.cosh((10*cycle)/ N ))) + TN

        # Cooling schedule 8
        elif cooling == 8:
            A = (1 / N) * math.log(T0 / TN)
            T = T0 * math.exp(-A*cycle)

        # Cooling schedule 9
        elif cooling == 9:
            A = (1 / (N*N)) * math.log(T0 / TN)
            T = T0 * math.exp(-A*(cycle*cycle))

        # Cooling schedule ?
        elif cooling == 10:
            T = T * alpha

        # Cooling schedule ?
        else:
            T = T * 0.8

        if cycle == N*1.1:
            break

    draw.draw(s_better, lastBetterCycle, lastBetterTemp, False)
    return s_better, iterations, distances, temp

def plotting(xIterations, yDistance, name):
    
    plt.figure()
    plt.plot(xIterations, yDistance)
    plt.xlabel("Iterations")
    plt.ylabel("Distance")
    plt.title(f"Cooling {name}")
    
    plt.annotate('%0.2f' % yDistance[-1], xy=(1, yDistance[-1]), xytext=(8, 0),
                    xycoords=('axes fraction', 'data'), textcoords='offset points')

    plt.savefig(f"convergence {name}.png")

    plt.close('all')


def boxplot(allDistances, allNames):
    
    plt.figure(figsize=(16,8))
    plt.ylabel("Distance")
    plt.title("All cooling profiles")
    plt.boxplot(allDistances, labels=allNames)
    # maybe turn of rotation if too full
    plt.xticks(rotation=10)

    plt.savefig("boxplot.png")

    plt.close('all')
    

def main():

    pygame.init()
    clock = pygame.time.Clock()
    pygame.display.set_caption(TITLE)

    cities_counter, max_x, max_y = len_cities(INSTANCE)
    # maybe change it
    
    cell_height = 850 / max_y
    cell_width = 1400 / max_x
    HEIGHT = max_y * cell_height + 80
    WIDTH = max_x * cell_width + 80
    WINDOW = pygame.display.set_mode((WIDTH, HEIGHT))

    cities = read_file(INSTANCE)
    generate_path(cities)
    path = make_path(cities)
    drawing = Draw(WINDOW, cell_width, cell_height)

    # while True:
    events()
    drawing.draw(path, 0, -1, False)
    pygame.display.update()
    
    allDistances = []
    allLastDistances = []
    allNames = []
    allNames_ = []
    allAVG = []
    allSTD = []

    pygame.image.save(WINDOW, f"initial_path.png")

    # going throught all cooling profile
    profiles = [7, 1, 9, 8, 5, 6, 2, 10, 11]
    profiles = [7, 1, 9, 8, 2]
    max_samples = 10
    # for i in range(12):
    for i in profiles:
        for j in range(max_samples):
            pygame.display.set_caption(f"{TITLE} --- profile {i} at {j+1} time")
            
            global T0_INITIAL_TEMPERATURE
            T0_INITIAL_TEMPERATURE = 50

            while sa(True, ALPHA_FACTOR, THERMAL_BALANCE, T0_INITIAL_TEMPERATURE, TN_FINAL_TEMPERATURE, N_COOLING_CYCLES, i, path, drawing):
                continue
            
            s_better, iterations, distances, temp = sa(False, ALPHA_FACTOR, THERMAL_BALANCE, T0_INITIAL_TEMPERATURE, TN_FINAL_TEMPERATURE, N_COOLING_CYCLES, i, path, drawing)
            
            if get_distance(s_better) < 440:
                print()
                print(get_distance(s_better), i, j+1)
                print()

            pygame.image.save(WINDOW, f"profile {i} {j+1} times path.png")
            plotting(iterations, distances, f"profile {i} {j+1} times")
            plotting([k for k in range(len(temp))], temp, f"temps {i} {j+1} times")
            
            allDistances.append(distances)
            allLastDistances.append(distances[-1])

            #filtering for boxplot
            if i == 10:
                if j == 9:
                    allNames.append(f"X\nX")
                elif j == 10:
                    allNames.append(f"X\nXI")
                else:
                    allNames.append(f"X\n{j+1}")
            elif i == 11:
                if j == 9:
                    allNames.append(f"XI\nX")
                elif j == 10:
                    allNames.append(f"XI\nXI")
                else:
                    allNames.append(f"XI\n{j+1}")
            else:
                if j == 9:
                    allNames.append(f"{i}\nX")
                elif j == 10:
                    allNames.append(f"{i}\nXI")
                else:
                    allNames.append(f"{i}\n{j+1}")

            allNames_.append(f"{j+1} times of profile {i}")
            allAVG.append(round(numpy.average(distances), 1))
            allSTD.append(round(numpy.std(distances), 1))

    boxplot(allDistances, allNames)
    
    col_names = ["Cooling Profile by Run", "minimum", "Average", "σ"]
    column = [allNames_, allLastDistances, allAVG, allSTD]

    table = BeautifulTable()
    for i in range(len(column)):
        table.columns.insert(i, column[i], header=col_names[i])
    table.set_style(BeautifulTable.STYLE_BOX_ROUNDED)
    print(table)

    print("ok")
main()
        

    