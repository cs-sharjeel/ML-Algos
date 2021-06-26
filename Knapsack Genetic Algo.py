import pyeasyga
data =[{'name': 'box1','value':4,'weight':12},
       {'name': 'box2','value':2,'weight':1},
       {'name': 'box3','value':10,'weight':4},
       {'name': 'box4','value':1,'weight':1},
       {'name': 'box5','value':2,'weight':2},]

ga = pyeasyga.GeneticAlgorithm(data)

def fitness(individual, data):
    values,weight=0,0
    for selected, box in zip(individual, data):
        if selected:
            values +=box.get('value')
            weights +=weight.get('weight')
        
    if weights >15 :
        values = 0
        return values
    
ga.fitness_function=fitness
ga.run()
print(ga.best_individual())