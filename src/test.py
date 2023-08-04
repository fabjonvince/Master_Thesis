from tools import find_kg_pathes


kg = [
    # first level
    ('A', 'B', 'C'),
    ('A', 'B', 'D'),
    ('A', 'B', 'E'),
    # second level C
    ('C', 'F', 'G'),
    ('C', 'F', 'H'),
    #second level D
    ('D', 'I', 'J'),
    ('D', 'I', 'K'),
    #second level E
    ('E', 'L', 'M'),
    ('E', 'L', 'N'),
    #third level G
    ('G', 'O', 'P'),
    ('G', 'O', 'Q'),
]

print(find_kg_pathes('A', 'C', kg, 5))
print(find_kg_pathes('A', 'Q', kg, 4))
print(find_kg_pathes('A', 'Q', kg, 3))
print(find_kg_pathes('A', 'Q', kg, 2))
print(find_kg_pathes('A', 'Q', kg, 1))
print(find_kg_pathes('A', 'P', kg, 4))
print(find_kg_pathes('A', 'J', kg, 4))
print(find_kg_pathes('A', 'ZZ', kg, 5))



