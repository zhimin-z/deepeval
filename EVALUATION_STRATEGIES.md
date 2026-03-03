# DeepEval Evaluation Workflow Strategy Support

This document identifies which strategies from the unified evaluation workflow are supported by DeepEval.

---

## **Phase 0: Provisioning (The Runtime)**

### **Step A: Harness Installation**

- ✅ **Strategy 1: Git Clone** - **SUPPORTED**
  - As an open-source project, DeepEval can be cloned and installed from source
  - Repository available at: https://github.com/confident-ai/deepeval
  - Manual installation for bleeding-edge versions or development work

- ✅ **Strategy 2: PyPI Packages** - **SUPPORTED**
  - DeepEval can be installed via `pip install -U deepeval`
  - Documentation: README.md, docs/docs/cli.mdx, docs/docs/getting-started.mdx
  - Additional dependencies can be installed via pip (e.g., `pip install chromadb langchain-core` for document synthesis)
  - Supports requirements files and git-based installations

- ❌ **Strategy 3: Node Package** - **NOT SUPPORTED**
  - DeepEval is Python-based only
  - No npm, npx, or Homebrew packages available

- ❌ **Strategy 4: Binary Packages** - **NOT SUPPORTED**
  - No standalone executable binaries available

- ❌ **Strategy 5: Container Images** - **NOT SUPPORTED**
  - No evidence of Docker or container image distribution found in documentation

### **Step B: Service Authentication**

- ✅ **Strategy 1: Evaluation Platform Authentication** - **SUPPORTED**
  - Authenticates with Confident AI evaluation platform to access platform services and features
  - Authentication via CLI: `deepeval login` (account registration and command-line login flows)
  - Enables full platform capabilities:
    - **Configuring evaluations**: Dataset management (push/pull), metric fine-tuning, annotation
    - **Running experiments**: Benchmark comparisons, hyperparameter tracking, A/B testing
    - **Viewing results**: Test run reports via `deepeval view`, dashboards, LLM traces
    - **Submitting to leaderboards**: Automatic test result uploads and sharing
  - Documentation: README.md (lines 28, 114-120, 350-376), docs/docs/cli.mdx (lines 60-65), docs/docs/getting-started.mdx
  - Platform features span multiple workflow phases (dataset prep, execution, reporting)

- ✅ **Strategy 2: API Provider Authentication** - **SUPPORTED**
  - Supports multiple LLM providers via API keys:
    - OpenAI (OPENAI_API_KEY)
    - Anthropic
    - Gemini
    - Azure OpenAI
    - Amazon Bedrock
    - Grok
    - Vertex AI
    - Ollama (local)
  - Configuration via environment variables or .env files
  - Documentation: README.md (lines 191-195, 337-393), docs/docs/cli.mdx, docs/integrations/models/
  - CLI commands: `deepeval set-*` and `--save=dotenv` for persistence

- ✅ **Strategy 3: Repository Authentication** - **SUPPORTED**
  - For accessing model repositories (e.g., Hugging Face for benchmarking)
  - Example in benchmarks-introduction.mdx shows loading models from Hugging Face
  - Required for custom model implementations and benchmark evaluations

---

## **Phase I: Specification (The Contract)**

### **Step A: SUT Preparation**

- ✅ **Strategy 1: Model-as-a-Service (Remote Inference)** - **SUPPORTED**
  - Supports remote API-based models:
    - OpenAI models (GPT-4, GPT-3.5, etc.)
    - Anthropic Claude
    - Google Gemini
    - Azure OpenAI
    - Amazon Bedrock
    - Grok
    - Vertex AI
  - Documentation: docs/docs/getting-started-rag.mdx (lines 51-162), docs/integrations/models/
  - Used for both evaluation and as LLM-as-judge in metrics

- ✅ **Strategy 2: Model-in-Process (Local Inference)** - **SUPPORTED**
  - Custom model implementation via `DeepEvalBaseLLM` class
  - Supports loading local models (e.g., via Hugging Face transformers, Ollama)
  - Example with Mistral 7B in docs/docs/benchmarks-introduction.mdx (lines 62-111)
  - Documentation: docs/docs/benchmarks-introduction.mdx, docs/integrations/models/ollama.mdx
  - Allows direct access to model weights and inference

- ❌ **Strategy 3: Algorithm Implementation (In-Memory Structures)** - **NOT SUPPORTED**
  - DeepEval focuses on LLM evaluation, not vector indexes or knowledge graph embeddings
  - No support for ANN algorithms, BM25 indexes, etc. as primary evaluation targets

- ⚠️ **Strategy 4: Policy/Agent Instantiation (Stateful Controllers)** - **PARTIALLY SUPPORTED**
  - Supports evaluating LLM-based AI agents with metrics like Task Completion, Tool Correctness
  - Documentation: docs/docs/metrics-introduction.mdx (lines 56-67)
  - Agentic metrics: Task Completion, Argument Correctness, Tool Correctness, Step Efficiency, Plan Adherence, Plan Quality
  - Does NOT support RL policies, robot controllers, or simulation-based agents
  - Focus is on LLM-based agents with tool use and reasoning capabilities

### **Step B: Benchmark Preparation (Inputs)**

- ✅ **Strategy 1: Benchmark Dataset Preparation (Offline)** - **SUPPORTED**
  - Pre-existing benchmark datasets:
    - MMLU (Massive Multitask Language Understanding)
    - HellaSwag
    - DROP
    - BIG-Bench Hard
    - TruthfulQA
    - HumanEval
    - GSM8K
    - and more (ARC, BBQ, BoolQ, IfEval, Lambada, LogiQA, MathQA, SQUAD, Winogrande)
  - Documentation: docs/docs/benchmarks-introduction.mdx, individual benchmark docs
  - Custom dataset creation via `EvaluationDataset` class
  - Dataset loading from Confident AI: `dataset.pull(alias="My Dataset")`
  - Documentation: docs/docs/evaluation-datasets.mdx (lines 136-143)

- ✅ **Strategy 2: Synthetic Data Generation (Generative)** - **SUPPORTED**
  - Synthesizer for generating synthetic test inputs and reference outputs
  - Generates `input` (test queries/scenarios) and optionally `expected_output` (reference labels)
  - Does NOT generate `actual_output` (which comes from the system under test)
  - Generate goldens from documents: `synthesizer.generate_goldens_from_docs()`
  - Generate goldens from scratch: `synthesizer.generate_goldens_from_scratch()`
  - Generate conversational goldens (scenario + expected_outcome) for multi-turn evaluation
  - Documentation: README.md (lines 122-140), docs/docs/synthesizer-introduction.mdx, docs/docs/synthesizer-generate-from-docs.mdx (lines 78, 98)

- ❌ **Strategy 3: Simulation Environment Setup (Simulated)** - **NOT SUPPORTED**
  - No support for 3D virtual environments, physics simulation, or multi-agent scenarios
  - DeepEval is focused on text-based LLM evaluation

- ✅ **Strategy 4: Production Traffic Sampling (Online)** - **SUPPORTED**
  - Tracing capabilities allow monitoring production LLM responses
  - Documentation: docs/docs/evaluation-llm-tracing.mdx, README.md (line 119)
  - "Monitor & evaluate LLM responses in product to improve datasets with real-world data"
  - Can collect real-world data via tracing for later evaluation
  - Integration with Confident AI platform for production monitoring

### **Step C: Benchmark Preparation (References)**

- ✅ **Strategy 1: Judge Preparation** - **SUPPORTED**
  - LLM-as-judge models for evaluation:
    - G-Eval metric
    - DAG (Deep Acyclic Graph) metric
    - Conversational G-Eval
    - Arena G-Eval for comparing multiple models
  - Pre-configured judge models using OpenAI, Anthropic, Gemini, etc.
  - Custom judge configuration via model selection
  - Documentation: docs/docs/metrics-introduction.mdx (lines 15-150), docs/docs/metrics-llm-evals.mdx

- ✅ **Strategy 2: Ground Truth Preparation** - **SUPPORTED**
  - Expected outputs in test cases (`expected_output` parameter)
  - Retrieval contexts for RAG evaluation (`retrieval_context` parameter)
  - Human annotations via Confident AI platform
  - Golden datasets with pre-defined reference materials
  - Documentation: docs/docs/evaluation-test-cases.mdx, docs/docs/evaluation-datasets.mdx

---

## **Phase II: Execution (The Run)**

### **Step A: SUT Invocation**

- ✅ **Strategy 1: Batch Inference** - **SUPPORTED**
  - Evaluate multiple test cases in a dataset
  - Parallel execution via `run_async=True` or `async_config`
  - Batch generation for benchmarks: `benchmark.evaluate(model=mistral_7b, batch_size=5)`
  - Documentation: README.md (lines 292-323), docs/docs/evaluation-introduction.mdx (lines 192-215), docs/docs/benchmarks-introduction.mdx (lines 129-144)
  - CLI command: `deepeval test run test_example.py -n 4` (parallel execution)

- ✅ **Strategy 2: Interactive Loop** - **SUPPORTED**
  - Multi-turn conversational evaluation via `ConversationalTestCase`
  - Turn-based evaluation with `Turn` objects
  - Support for agentic workflows with tool use
  - Documentation: docs/docs/getting-started-rag.mdx (lines 540-616), docs/docs/evaluation-multiturn-test-cases.mdx
  - Metrics: Task Completion, Tool Correctness, Argument Correctness for agents

- ✅ **Strategy 3: Arena Battle** - **SUPPORTED**
  - `ArenaTestCase` for comparing multiple models on same input
  - `Contestant` class for each model variant
  - Arena G-Eval metric for pairwise comparison
  - Documentation: docs/docs/evaluation-arena-test-cases.mdx (lines 1-100), docs/docs/metrics-arena-g-eval.mdx
  - Example: comparing GPT-4, Claude-4, and Gemini-2.5 on same input

- ✅ **Strategy 4: Production Streaming** - **SUPPORTED**
  - LLM tracing for monitoring production execution
  - `@observe` decorator for continuous monitoring
  - Documentation: docs/docs/evaluation-llm-tracing.mdx, README.md (line 119)
  - Can evaluate components in production via tracing
  - Integration with Confident AI platform for production monitoring and real-time metric collection

---

## **Phase III: Assessment (The Score)**

### **Step A: Individual Scoring**

- ✅ **Strategy 1: Deterministic Measurement** - **SUPPORTED**
  - Exact Match metric
  - Token-based metrics via custom metric implementation
  - JSON Correctness metric for structured outputs
  - Documentation: docs/docs/metrics-exact-match.mdx, docs/docs/metrics-json-correctness.mdx
  - Can implement custom deterministic metrics (BLEU, ROUGE) via `BaseMetric` class

- ❌ **Strategy 2: Embedding Measurement** - **NOT SUPPORTED**
  - DeepEval does not use embedding-based similarity calculations (BERTScore, sentence embeddings, cosine similarity)
  - All similarity judgments are done via LLM-as-judge (subjective measurement)
  - Metrics like Answer Relevancy use LLM prompts, not embedding comparisons
  - No support for neural similarity models like COMET or BERTScore

- ✅ **Strategy 3: Subjective Measurement** - **SUPPORTED**
  - LLM-as-judge metrics:
    - G-Eval (custom criteria evaluation)
    - DAG (Deep Acyclic Graph)
    - Conversational G-Eval
    - Arena G-Eval
  - Pre-built subjective metrics:
    - Bias
    - Toxicity
    - Hallucination
    - Faithfulness
    - Summarization
  - Documentation: docs/docs/metrics-introduction.mdx (lines 15-37), docs/docs/metrics-llm-evals.mdx
  - All metrics output score (0-1) with reasoning

- ⚠️ **Strategy 4: Performance Measurement** - **PARTIALLY SUPPORTED**
  - Step Efficiency metric evaluates resource economy (minimizing tool calls, LLM invocations, reasoning steps)
  - Documentation: deepeval/metrics/step_efficiency/step_efficiency.py
  - However, NO support for:
    - Latency measurement (wall-clock time, response time)
    - Throughput measurement (requests per second, tokens per second)
    - Memory consumption (RAM usage, GPU memory)
    - Energy consumption (power usage, carbon footprint)
  - Focus remains on reasoning efficiency (quality), not runtime performance

### **Step B: Collective Aggregation**

- ✅ **Strategy 1: Score Aggregation** - **SUPPORTED**
  - Aggregate metrics across test cases in a dataset
  - Overall scores for benchmarks
  - Test run summaries with pass/fail counts
  - Documentation: docs/docs/benchmarks-introduction.mdx (lines 146-150), docs/docs/evaluation-introduction.mdx
  - Hyperparameter tracking for comparing iterations

- ❌ **Strategy 2: Uncertainty Quantification** - **NOT SUPPORTED**
  - No evidence of bootstrap resampling or confidence intervals
  - No PPI (Prediction-Powered Inference) implementation mentioned

---

## **Phase IV: Reporting (The Output)**

### **Step A: Insight Presentation**

- ✅ **Strategy 1: Execution Tracing** - **SUPPORTED**
  - LLM tracing shows step-by-step execution logs
  - Spans for individual components (LLM calls, retrievers, tools)
  - Trace visualization on Confident AI platform
  - Documentation: docs/docs/evaluation-llm-tracing.mdx (lines 1-100)
  - `@observe` decorator for tracing components

- ✅ **Strategy 2: Subgroup Analysis** - **SUPPORTED**
  - Benchmark task-specific scores available via `task_scores` attribute
  - Can analyze performance by task category in benchmarks
  - Documentation: docs/docs/benchmarks-introduction.mdx (lines 158-173)
  - Task-level breakdown in pandas DataFrame format
  - Individual prediction details for fine-grained analysis

- ✅ **Strategy 3: Chart Generation** - **SUPPORTED**
  - Visual reports on Confident AI platform
  - Testing reports with metric visualizations
  - Documentation: README.md (lines 63, 377), docs/docs/metrics-introduction.mdx (lines 142-148)
  - Confident AI provides visual representations of evaluation results

- ✅ **Strategy 4: Dashboard Creation** - **SUPPORTED**
  - Confident AI platform provides web-based dashboards
  - Interactive metric comparisons
  - Test run history and comparisons
  - Documentation: README.md (lines 350-376), CLI commands like `deepeval view`
  - Shareable testing reports URL

- ✅ **Strategy 5: Leaderboard Publication** - **SUPPORTED**
  - Integration with Confident AI for result sharing
  - Public/private test run sharing via URLs
  - Benchmark results can be uploaded and compared
  - Documentation: README.md (lines 114-120, 350-376)
  - `deepeval login` and automatic upload on evaluation

- ✅ **Strategy 6: Regression Alerting** - **SUPPORTED**
  - Regression testing capabilities via Confident AI platform
  - Documentation: docs/docs/evaluation-unit-testing.mdx (lines 12, 18), docs/docs/getting-started.mdx
  - "Automate regression testing" and "catch regressions" features
  - A/B regression testing dashboard showing improvements (green) and regressions (red)
  - Test run comparison to detect performance degradation

---

## Summary

### Supported Strategies by Phase:

**Phase 0: Provisioning**
- 2/5 installation strategies (Git Clone, PyPI)
- 3/3 authentication strategies (Platform, API Provider, Repository)

**Phase I: Specification**
- 3/4 SUT preparation strategies (Remote Inference, Local Inference, LLM-based Agents - partially)
- 3/4 benchmark preparation strategies (Offline Datasets, Synthetic Generation, Production Traffic Sampling)
- 2/2 reference preparation strategies (Judge Preparation, Ground Truth)

**Phase II: Execution**
- 4/4 invocation strategies (Batch Inference, Interactive Loop, Arena Battle, Production Streaming)

**Phase III: Assessment**
- 2/4 individual scoring strategies (Deterministic, Subjective only)
- 1/4 partial support (Performance - Step Efficiency for resource economy only)
- 1/2 aggregation strategies (Score Aggregation only)

**Phase IV: Reporting**
- 6/6 presentation strategies (all strategies supported including Regression Alerting)

### Overall Coverage:
- **Total Strategies in Framework**: 34
- **Fully Supported**: 21
- **Partially Supported**: 3 (LLM-based agents, Step Efficiency for resource economy, Production Traffic)
- **Not Supported**: 10
- **Support Rate**: ~62% (21/34) or ~71% (24/34 including partial support)

### Key Strengths:
1. Comprehensive LLM evaluation metric library (50+ metrics)
2. Strong integration with Confident AI platform for visualization and collaboration
3. Multiple LLM provider support (OpenAI, Anthropic, Gemini, etc.)
4. Benchmark dataset library (MMLU, HellaSwag, HumanEval, etc.)
5. Synthetic data generation capabilities
6. Multi-turn and conversational evaluation support
7. Arena-style model comparison
8. LLM tracing for component-level evaluation
9. Production monitoring and regression testing via Confident AI platform

### Key Gaps:
1. No containerized deployment option
2. No embedding-based similarity metrics (BERTScore, cosine similarity, COMET)
3. No runtime performance metrics (latency, throughput, memory usage, energy)
4. No simulation environment support (3D, physics, RL)
5. No uncertainty quantification (confidence intervals, bootstrap)
6. Limited support for non-LLM evaluation targets (e.g., vector indexes, graph embeddings)
7. Agent evaluation limited to LLM-based agents (not RL policies or robot controllers)
8. No binary package distribution

---

*This analysis is based on the DeepEval documentation as of the repository state on 2025-12-12.*
