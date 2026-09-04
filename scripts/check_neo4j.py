from dotenv import load_dotenv
load_dotenv()

from graph.neo4j_infrastructure import InfrastructureGraph

if __name__ == "__main__":
    graph = InfrastructureGraph()
    try:
        graph.verify()
        graph.initialize()
        print("Neo4j AuraDB connectivity and constraints OK.")
    finally:
        graph.close()
