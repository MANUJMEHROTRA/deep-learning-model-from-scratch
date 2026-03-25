```mermaid
flowchart TB
    classDef k8s fill:#326ce5,stroke:#fff,stroke-width:2px,color:#fff,rx:5px,ry:5px;
    classDef agent fill:#f5f5dc,stroke:#333,stroke-width:2px,rx:5px,ry:5px;
    classDef db fill:#ffebcd,stroke:#333,stroke-width:2px,rx:5px,ry:5px;
    classDef obs fill:#e6e6fa,stroke:#333,stroke-width:2px,rx:5px,ry:5px;
    classDef tool fill:#d1e8e2,stroke:#333,stroke-width:2px,rx:5px,ry:5px;

    User((User / Client))

    subgraph Kubernetes_Cluster [Kubernetes Cluster]
        direction TB
        Ingress[Ingress Controller\nLoad Balancer]:::k8s

        subgraph App_Tier [FastAPI + Agent Tier Auto-Scaled via HPA]
            direction LR
            Pod1[FastAPI App Pod 1\n+ LangGraph Worker]
            Pod2[FastAPI App Pod 2\n+ LangGraph Worker]
            PodN[FastAPI App Pod N\n...]
        end

        subgraph State_Tier [State & Memory Tier]
            Redis[(Redis Cluster\nCheckpointer & Cache)]:::db
        end

        subgraph Observability_Stack [Observability & Monitoring]
            OTel[OpenTelemetry\nCollector]:::obs
            Prom[Prometheus]:::obs
            Grafana[Grafana]:::obs
            
            OTel --> Prom --> Grafana
        end

        Ingress -- "HTTP Queries" --> App_Tier
        App_Tier -- "Read/Write Agent State" --> Redis
        App_Tier -. "Traces, Logs & Metrics" .-> OTel
    end

    subgraph Agentic_Workflow [LangGraph Core Logic]
        direction TB
        Graph[LangGraph StateGraph\n(Nodes & Edges)]:::agent
        LLM[LLM Provider\n(OpenAI/Gemini/Anthropic)]:::agent
        MCPClient[MCP Client]:::agent

        Graph <-->|Prompts & Generation| LLM
        Graph <-->|Tool Execution Requests| MCPClient
    end

    subgraph External_Tools [MCP Servers & Tools]
        MCPS1[MCP Server 1\n(Internal Data)]:::tool
        MCPS2[MCP Server 2\n(Web APIs)]:::tool
        Tool1[(Vector DB / RAG)]:::tool
        Tool2[Web Scraper / API]:::tool

        MCPS1 --> Tool1
        MCPS2 --> Tool2
    end

    %% Connections linking the conceptual blocks
    User -- "Sends Query" --> Ingress
    Pod1 -. "Executes" .-> Agentic_Workflow
    MCPClient <-->|Model Context Protocol| MCPS1
    MCPClient <-->|Model Context Protocol| MCPS2
