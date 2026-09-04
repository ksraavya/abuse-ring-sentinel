from dotenv import load_dotenv
load_dotenv()

from graph.neo4j_infrastructure import InfrastructureGraph

if __name__ == "__main__":
    graph = InfrastructureGraph()
    try:
        graph.verify()
        graph.reset()
        graph.initialize()
        print("Neo4j AuraDB reset and initialized.")
    finally:
        graph.close()
