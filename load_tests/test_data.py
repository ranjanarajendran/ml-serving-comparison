"""
Test data for load testing.

Provides sample texts of varying lengths for embedding benchmarks.
"""

# Short texts (10-50 words)
SHORT_TEXTS = [
    "Machine learning is transforming how we build software.",
    "Natural language processing enables computers to understand human language.",
    "Deep learning models require large amounts of training data.",
    "Text embeddings capture semantic meaning in vector form.",
    "API performance is critical for production ML systems.",
]

# Medium texts (50-150 words)
MEDIUM_TEXTS = [
    """
    Machine learning operations, or MLOps, is a set of practices that aims to deploy
    and maintain machine learning models in production reliably and efficiently.
    The practice combines machine learning, DevOps, and data engineering. MLOps seeks
    to increase automation and improve the quality of production ML while also focusing
    on business and regulatory requirements. While DevOps brings together software
    development and operations, MLOps extends this to include the experimental nature
    of data science and machine learning.
    """,
    """
    REST (Representational State Transfer) is an architectural style for designing
    networked applications. It relies on a stateless, client-server protocol, almost
    always HTTP. RESTful applications use HTTP requests to post data, read data, and
    delete data. REST uses standard HTTP methods like GET, POST, PUT, and DELETE.
    The key abstraction of information in REST is a resource, and any information that
    can be named can be a resource. REST is designed to use a stateless communication
    protocol, typically HTTP.
    """,
    """
    gRPC is a modern open source high performance Remote Procedure Call framework that
    can run in any environment. It uses HTTP/2 for transport, Protocol Buffers as the
    interface description language, and provides features such as authentication,
    bidirectional streaming and flow control, blocking or nonblocking bindings, and
    cancellation and timeouts. gRPC is based on the idea of defining a service,
    specifying the methods that can be called remotely with their parameters and return types.
    """,
]

# Long texts (150-300 words)
LONG_TEXTS = [
    """
    GraphQL is a query language for APIs and a runtime for fulfilling those queries
    with your existing data. GraphQL provides a complete and understandable description
    of the data in your API, gives clients the power to ask for exactly what they need
    and nothing more, makes it easier to evolve APIs over time, and enables powerful
    developer tools. Unlike REST, where you might need multiple endpoints to fetch
    related data, GraphQL allows you to get all the data your app needs in a single
    request. GraphQL queries access not just the properties of one resource but also
    smoothly follow references between them. While typical REST APIs require loading
    from multiple URLs, GraphQL APIs get all the data your app needs in a single request.
    Apps using GraphQL can be quick even on slow mobile network connections. GraphQL
    APIs are organized in terms of types and fields, not endpoints. Access the full
    capabilities of your data from a single endpoint. GraphQL uses types to ensure
    Apps only ask for what's possible and provide clear and helpful errors.
    """,
    """
    HTTP/2 is a major revision of the HTTP network protocol. The primary goals for
    HTTP/2 are to reduce latency by enabling full request and response multiplexing,
    minimize protocol overhead via efficient compression of HTTP header fields, and
    add support for request prioritization and server push. HTTP/2 does not modify
    the application semantics of HTTP in any way. All the core concepts found in
    HTTP/1.1, such as HTTP methods, status codes, URIs, and header fields, remain
    in place. Instead, HTTP/2 modifies how the data is formatted and transported
    between the client and server, both of which manage the entire process, and hides
    all the complexity from our applications within the new framing layer. The binary
    framing layer is not backward compatible with HTTP/1.x servers or clients, hence
    the major protocol version increment to HTTP/2. Key features include binary protocol,
    multiplexing, header compression, and server push capabilities.
    """,
]

# Technical texts about embeddings and ML
TECHNICAL_TEXTS = [
    """
    Sentence embeddings are vector representations of sentences that capture their
    semantic meaning. Modern sentence embedding models like SentenceTransformers use
    pre-trained language models such as BERT, RoBERTa, or MPNet as a base and apply
    techniques like mean pooling or CLS token extraction to generate fixed-size vectors.
    These embeddings enable semantic similarity search, clustering, and classification
    tasks. The all-MiniLM-L6-v2 model produces 384-dimensional embeddings and offers
    an excellent balance between performance and computational efficiency.
    """,
    """
    Protocol Buffers, or protobuf, is Google's language-neutral, platform-neutral
    extensible mechanism for serializing structured data. It's like JSON or XML, but
    smaller, faster, and simpler. You define how you want your data to be structured
    once, then you can use special generated source code to easily write and read
    your structured data to and from a variety of data streams using a variety of
    languages. Protocol Buffers are particularly useful in RPC systems and for data
    storage formats where efficiency is critical.
    """,
]

def get_random_text(length='medium'):
    """
    Get a random text of specified length.

    Args:
        length: 'short', 'medium', 'long', or 'technical'

    Returns:
        Random text string
    """
    import random

    if length == 'short':
        return random.choice(SHORT_TEXTS)
    elif length == 'medium':
        return random.choice(MEDIUM_TEXTS)
    elif length == 'long':
        return random.choice(LONG_TEXTS)
    elif length == 'technical':
        return random.choice(TECHNICAL_TEXTS)
    else:
        # Mix of all types
        all_texts = SHORT_TEXTS + MEDIUM_TEXTS + LONG_TEXTS + TECHNICAL_TEXTS
        return random.choice(all_texts)


def get_batch_texts(count=5, length='medium'):
    """
    Get multiple texts for batch testing.

    Args:
        count: Number of texts to return
        length: 'short', 'medium', 'long', 'technical', or 'mixed'

    Returns:
        List of text strings
    """
    return [get_random_text(length) for _ in range(count)]
