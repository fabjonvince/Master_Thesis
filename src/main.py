import argparse
from preprocess import text_to_graph, print_triplets


def main(args):
    print("In Main")
    sent1 = "Where is born Barack Obama?"
    sent2 = "Obama was born in Honolulu, Hawaii. After graduating from Columbia University in 1983, he worked as a community organizer in Chicago. In 1988, he enrolled in Harvard Law School, where he was the first black president of the Harvard Law Review."
    graph = text_to_graph(3, sent1, sent2)
    print_triplets(graph)

if __name__ == '__main__':
    main(None)