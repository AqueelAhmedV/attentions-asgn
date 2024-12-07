from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from neo4j import GraphDatabase
from ollama_utils import OllamaEmbedder, OllamaLLM
from neo4j_graphrag.generation import GraphRAG
from neo4j_graphrag.retrievers import Text2CypherRetriever
from neo4j_graphrag.experimental.components.entity_relation_extractor import (
    LLMEntityRelationExtractor,
    OnError
)
from neo4j_graphrag.experimental.components.schema import (
    SchemaBuilder,
    SchemaEntity,
    SchemaProperty,
    SchemaRelation,
)
from neo4j_graphrag.generation.prompts import RagTemplate
import hashlib
import secrets
from config import OLLAMA_BASE_URL, DEFAULT_MODEL, NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD

class MemoryAgent:
    def __init__(self):
        """Initialize Memory Agent with Neo4j and Ollama configurations"""
        self.driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        
        # Initialize Ollama LLM
        self.llm = OllamaLLM(
            model_name=DEFAULT_MODEL,
            model_params={
                "temperature": 0.4,
                "max_tokens": 10000,
            },
            base_url=OLLAMA_BASE_URL
        )
        
        # Initialize Ollama embeddings
        self.embedder = OllamaEmbedder(
            model=DEFAULT_MODEL,
            base_url=OLLAMA_BASE_URL
        )
        
        self.user_sessions: Dict[str, datetime] = {}
        self.session_timeout = timedelta(hours=24)
        
        # Remove these calls from __init__
        # self._init_schema()
        # self._init_retrievers()

    async def initialize(self):
        """Asynchronously initialize the schema and retrievers"""
        await self._init_schema()
        self._init_retrievers()

    async def _init_schema(self):
        """Initialize the knowledge graph schema"""
        schema_builder = SchemaBuilder()
        schema_result = await schema_builder.run(
            entities=[
                SchemaEntity(
                    label="User",
                    properties=[
                        SchemaProperty(name="username", type="STRING"),
                        SchemaProperty(name="last_active", type="LOCAL_DATETIME"),
                        SchemaProperty(name="password", type="STRING"),
                    ],
                ),
                SchemaEntity(
                    label="Preference",
                    properties=[
                        SchemaProperty(name="category", type="STRING"),
                        SchemaProperty(name="value", type="STRING"),
                        SchemaProperty(name="timestamp", type="LOCAL_DATETIME"),
                    ],
                ),
                SchemaEntity(
                    label="City",
                    properties=[
                        SchemaProperty(name="name", type="STRING"),
                        SchemaProperty(name="country", type="STRING"),
                    ],
                ),
                SchemaEntity(
                    label="Activity",
                    properties=[
                        SchemaProperty(name="type", type="STRING"),
                        SchemaProperty(name="description", type="STRING"),
                    ],
                ),
            ],
            relations=[
                SchemaRelation(label="HAS_PREFERENCE"),
                SchemaRelation(label="VISITED"),
                SchemaRelation(label="INTERESTED_IN"),
                SchemaRelation(label="DISLIKES"),
            ],
            potential_schema=[
                ("User", "HAS_PREFERENCE", "Preference"),
                ("User", "VISITED", "City"),
                ("User", "INTERESTED_IN", "Activity"),
                ("User", "DISLIKES", "Activity"),
            ],
        )
        self.schema = schema_result

    def _init_retrievers(self):
        """Initialize the Text2Cypher retriever for memory queries"""
        # Print schema for debugging
        print("Schema type:", type(self.schema))
        print("Schema content:", self.schema)
        
        self.retriever = Text2CypherRetriever(
            driver=self.driver,
            llm=self.llm,
            neo4j_schema=str(self.schema),  # Just convert the schema object to string directly
            examples=[
                "USER INPUT: 'What are John's preferences?' QUERY: MATCH (u:User {username: 'john'})-[:HAS_PREFERENCE]->(p:Preference) RETURN p.category, p.value",
                "USER INPUT: 'Show me cities Mary visited' QUERY: MATCH (u:User {username: 'mary'})-[:VISITED]->(c:City) RETURN c.name, c.country",
                "USER INPUT: 'What activities does Tom like?' QUERY: MATCH (u:User {username: 'tom'})-[:INTERESTED_IN]->(a:Activity) RETURN a.type, a.description"
            ]
        )

    async def extract_preferences(self, username: str, text: str) -> List[Dict[str, Any]]:
        """Extract preferences and entities from user input"""
        extractor = LLMEntityRelationExtractor(
            llm=self.llm,
            on_error=OnError.RAISE
        )
        
        # Create a TextChunks object with our input
        from neo4j_graphrag.experimental.components.types import TextChunk, TextChunks
        chunks = TextChunks(chunks=[TextChunk(text=text, index=0)])
        
        # Process the text using the run method
        result = await extractor.run(
            chunks=chunks,
            schema=self.schema,
            examples=f"Extract preferences and interests for user {username}"
        )
        
        # Convert the Neo4jGraph result to our desired format
        extracted_data = []
        if result.nodes:  # Extract entities
            extracted_data.append({"entities": result.nodes})
        if result.relationships:  # Extract relationships
            extracted_data.append({"relationships": result.relationships})
        
        return extracted_data

    async def store_user_memory(self, username: str, text: str) -> None:
        """Store user memory in the graph database"""
        try:
            # Extract preferences using LLM
            extracted_data = await self.extract_preferences(username, text)
            
            # Initialize GraphRAG for question answering
            rag = GraphRAG(
                retriever=self.retriever,  # Remove embedder parameter
                llm=self.llm,
                prompt_template=RagTemplate()  # Optional: customize the template if needed
            )
            
            # Store the memory
            timestamp = datetime.now()
            
            # Create memory nodes and relationships
            query = """
            MATCH (u:User {username: $username})
            CREATE (m:Memory {text: $text, timestamp: $timestamp})
            CREATE (u)-[:REMEMBERS]->(m)
            """
            self.driver.execute_query(
                query,
                username=username,
                text=text,
                timestamp=timestamp
            )
            
            # Store extracted preferences if any
            if extracted_data:
                self._store_preferences(username, extracted_data)
                
        except Exception as e:
            print(f"Error storing memory: {e}")

    async def get_user_preferences(self, username: str) -> Dict[str, Any]:
        """Retrieve user preferences and memory"""
        query = f"""
        MATCH (u:User {{username: $username}})
        OPTIONAL MATCH (u)-[:HAS_PREFERENCE]->(p:Preference)
        OPTIONAL MATCH (u)-[:VISITED]->(c:City)
        OPTIONAL MATCH (u)-[:INTERESTED_IN]->(a:Activity)
        RETURN u, collect(distinct p) as preferences, 
               collect(distinct c) as visited_cities,
               collect(distinct a) as interests
        """
        result = await self.driver.execute_query(query, username=username)
        return self._format_preferences(result)

    def _format_preferences(self, result) -> Dict[str, Any]:
        """Format the Neo4j result into a structured dictionary"""
        if not result or not result[0]:
            return {}
        
        record = result[0]
        return {
            "preferences": [dict(p) for p in record["preferences"]],
            "visited_cities": [dict(c) for c in record["visited_cities"]],
            "interests": [dict(a) for a in record["interests"]]
        }

    async def close(self):
        """Close the Neo4j connection"""
        await self.driver.close()

    def _hash_password(self, password: str) -> str:
        """Hash a password for storing."""
        return hashlib.sha256(password.encode()).hexdigest()
    
    def create_user(self, username: str, password: str) -> bool:
        """Create a new user in the graph database"""
        hashed_password = self._hash_password(password)
        try:
            query = """
            MERGE (u:User {username: $username})
            ON CREATE SET u.password = $password,
                        u.created_at = datetime(),
                        u.last_active = datetime()
            ON MATCH SET u.last_active = datetime()
            RETURN u.username
            """
            result = self.driver.execute_query(
                query, 
                username=username, 
                password=hashed_password,
                database_="neo4j"
            )
            return bool(result[0]) if result else False
        except Exception as e:
            print(f"Error creating user: {e}")
            return False

    def authenticate_user(self, username: str, password: str) -> Optional[str]:
        """Authenticate user and return session token if successful"""
        hashed_password = self._hash_password(password)
        try:
            query = """
            MATCH (u:User {username: $username, password: $password})
            SET u.last_active = datetime()
            RETURN u.username
            """
            result = self.driver.execute_query(
                query, 
                username=username, 
                password=hashed_password,
                database_="neo4j"
            )
            
            if result and result[0]:
                # Generate session token
                session_token = secrets.token_urlsafe(32)
                self.user_sessions[session_token] = datetime.now()
                return session_token
            return None
        except Exception as e:
            print(f"Authentication error: {e}")
            return None

    def validate_session(self, session_token: str) -> bool:
        """Validate if a session token is valid and not expired"""
        if session_token not in self.user_sessions:
            return False
        
        session_time = self.user_sessions[session_token]
        if datetime.now() - session_time > self.session_timeout:
            del self.user_sessions[session_token]
            return False
        
        return True 

    def _store_preferences(self, username: str, extracted_data: List[Dict[str, Any]]) -> None:
        """Store extracted preferences and relationships in Neo4j"""
        try:
            # Process entities
            for data in extracted_data:
                if 'entities' in data:
                    for entity in data['entities']:
                        # Debug prints
                        print("\n--- Debug Info ---")
                        print(f"Username: {username}")
                        print(f"Entity properties: {entity.properties}")
                        
                        query = """
                        MATCH (u:User {username: $username}) 
                        CREATE (u)-[:INTERESTED_IN]->(n:Preference {text: $text})
                        """
                        
                        # Create parameters dict
                        params = {
                            "username": username,
                            "text": entity.properties["text"]
                        }
                        
                        print(f"Query: {query}")
                        print(f"Parameters: {params}")
                        
                        self.driver.execute_query(
                            query,
                            parameters=params,  # Pass parameters directly without unpacking
                            database_="neo4j"
                        )
                
                # Process relationships
                if 'relationships' in data:
                    for rel in data['relationships']:
                        print(f"Processing relationship: {rel}")
                
        except Exception as e:
            print(f"Error storing preferences: {str(e)}")
            print("Parameters that were attempted:", params)
            import traceback
            traceback.print_exc() 