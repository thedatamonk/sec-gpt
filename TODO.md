## Immediate ToDo

1. ~~Complete SecAgent code~~
2. ~~Implement all the tools - I am gonna use sec-edgar tool implementations the MCP is full of shit~~
3. Write unit tests for all the components
4. Integrate LLM system evaluation using DeepEval
    - Use the following repo - [financial dataset using LLM](https://github.com/virattt/financial-datasets) to create a "golden" dataset to evaluate the LLM.
    - Then use frameworks like DeepEval to evaluate the LLM agent on this "golden" dataset.
5. ~~Tooling specific changes - ~~
6. ~~Replanning strategy for agent~~
7. Implement caching, rate limiting
    - edgartools recommends caching responses to avoid breaching SEC rate limits - 10 requests per second is the rate limit
8. Add Key metrics use in each component for monitoring
    - token per second, other metrics
    - refer to the linkedin post that mentions different metrics required for judging the quality of an LLM
