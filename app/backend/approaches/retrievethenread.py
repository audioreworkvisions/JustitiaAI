from typing import Any, Optional

from azure.search.documents.aio import SearchClient
from azure.search.documents.models import VectorQuery
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam
from openai_messages_token_helper import build_messages, get_token_limit

from approaches.approach import Approach, ThoughtStep
from core.authentication import AuthenticationHelper


class RetrieveThenReadApproach(Approach):
    """
    Simple retrieve-then-read implementation, using the AI Search and OpenAI APIs directly. It first retrieves
    top documents from search, then constructs a prompt with them, and then uses OpenAI to generate a completion
    (answer) with that prompt.
    """

    system_chat_template = (
        "Du bist ein intelligenter Assistent namens Justitia AI und hilfst den Benutzern bei Fragen zu Schweizer Rechtsvorschriften wie dem Steuergesetz, Strafgesetz und Obligationenrecht. "
        + "Verwende 'du', um dich auf die Person zu beziehen, die die Frage stellt, auch wenn sie 'ich' verwendet. "
        + "Beantworte die folgende Frage nur mit den in den Quellen unten bereitgestellten Daten. "
        + "Jede Quelle hat einen Namen, gefolgt von einem Doppelpunkt und den eigentlichen Informationen. Nenne immer den Quellennamen für jede Tatsache, die du in der Antwort verwendest. "
        + "Wenn du nicht mit den unten stehenden Quellen antworten kannst, gib an, dass du es nicht weißt. Nutze das folgende Beispiel, um zu antworten."
    )

    # Beispielgespräch (shots/sample conversation)
    question = """
'Was sind die Voraussetzungen für eine Verurteilung nach dem Schweizer Strafgesetz?'

Quellen:
info1.txt: Die Voraussetzungen für eine Verurteilung nach dem Strafgesetzbuch beinhalten die Absicht, eine Straftat zu begehen, sowie den Nachweis der Tat und eines schuldhaften Verhaltens.
info2.pdf: Zu den Voraussetzungen gehören auch Fahrlässigkeit, wenn das Gesetz dies ausdrücklich vorsieht.
info3.pdf: Das Strafgesetzbuch regelt auch, dass gewisse Vergehen nur dann verfolgt werden, wenn ein Strafantrag gestellt wurde.
"""
    answer = "Die Voraussetzungen für eine Verurteilung nach dem Strafgesetzbuch beinhalten die Absicht, eine Straftat zu begehen, sowie den Nachweis der Tat und eines schuldhaften Verhaltens [info1.txt]. Fahrlässigkeit ist ebenfalls eine Voraussetzung, wenn das Gesetz dies vorsieht [info2.pdf]."

    def __init__(
        self,
        *,
        search_client: SearchClient,
        auth_helper: AuthenticationHelper,
        openai_client: AsyncOpenAI,
        chatgpt_model: str,
        chatgpt_deployment: Optional[str],  # Not needed for non-Azure OpenAI
        embedding_model: str,
        embedding_deployment: Optional[str],  # Not needed for non-Azure OpenAI or for retrieval_mode="text"
        embedding_dimensions: int,
        sourcepage_field: str,
        content_field: str,
        query_language: str,
        query_speller: str,
    ):
        self.search_client = search_client
        self.chatgpt_deployment = chatgpt_deployment
        self.openai_client = openai_client
        self.auth_helper = auth_helper
        self.chatgpt_model = chatgpt_model
        self.embedding_model = embedding_model
        self.embedding_dimensions = embedding_dimensions
        self.chatgpt_deployment = chatgpt_deployment
        self.embedding_deployment = embedding_deployment
        self.sourcepage_field = sourcepage_field
        self.content_field = content_field
        self.query_language = query_language
        self.query_speller = query_speller
        self.chatgpt_token_limit = get_token_limit(chatgpt_model)

    async def run(
        self,
        messages: list[ChatCompletionMessageParam],
        session_state: Any = None,
        context: dict[str, Any] = {},
    ) -> dict[str, Any]:
        q = messages[-1]["content"]
        if not isinstance(q, str):
            raise ValueError("The most recent message content must be a string.")
        overrides = context.get("overrides", {})
        seed = overrides.get("seed", None)
        auth_claims = context.get("auth_claims", {})
        use_text_search = overrides.get("retrieval_mode") in ["text", "hybrid", None]
        use_vector_search = overrides.get("retrieval_mode") in ["vectors", "hybrid", None]
        use_semantic_ranker = True if overrides.get("semantic_ranker") else False
        use_semantic_captions = True if overrides.get("semantic_captions") else False
        top = overrides.get("top", 3)
        minimum_search_score = overrides.get("minimum_search_score", 0.0)
        minimum_reranker_score = overrides.get("minimum_reranker_score", 0.0)
        filter = self.build_filter(overrides, auth_claims)

        # If retrieval mode includes vectors, compute an embedding for the query
        vectors: list[VectorQuery] = []
        if use_vector_search:
            vectors.append(await self.compute_text_embedding(q))

        results = await self.search(
            top,
            q,
            filter,
            vectors,
            use_text_search,
            use_vector_search,
            use_semantic_ranker,
            use_semantic_captions,
            minimum_search_score,
            minimum_reranker_score,
        )

        # Process results
        sources_content = self.get_sources_content(results, use_semantic_captions, use_image_citation=False)

        # Append user message
        content = "\n".join(sources_content)
        user_content = q + "\n" + f"Sources:\n {content}"

        response_token_limit = 1024
        updated_messages = build_messages(
            model=self.chatgpt_model,
            system_prompt=overrides.get("prompt_template", self.system_chat_template),
            few_shots=[{"role": "user", "content": self.question}, {"role": "assistant", "content": self.answer}],
            new_user_content=user_content,
            max_tokens=self.chatgpt_token_limit - response_token_limit,
        )

        chat_completion = await self.openai_client.chat.completions.create(
            # Azure OpenAI takes the deployment name as the model name
            model=self.chatgpt_deployment if self.chatgpt_deployment else self.chatgpt_model,
            messages=updated_messages,
            temperature=overrides.get("temperature", 0.3),
            max_tokens=response_token_limit,
            n=1,
            seed=seed,
        )

        data_points = {"text": sources_content}
        extra_info = {
            "data_points": data_points,
            "thoughts": [
                ThoughtStep(
                    "Search using user query",
                    q,
                    {
                        "use_semantic_captions": use_semantic_captions,
                        "use_semantic_ranker": use_semantic_ranker,
                        "top": top,
                        "filter": filter,
                        "use_vector_search": use_vector_search,
                        "use_text_search": use_text_search,
                    },
                ),
                ThoughtStep(
                    "Search results",
                    [result.serialize_for_results() for result in results],
                ),
                ThoughtStep(
                    "Prompt to generate answer",
                    updated_messages,
                    (
                        {"model": self.chatgpt_model, "deployment": self.chatgpt_deployment}
                        if self.chatgpt_deployment
                        else {"model": self.chatgpt_model}
                    ),
                ),
            ],
        }

        return {
            "message": {
                "content": chat_completion.choices[0].message.content,
                "role": chat_completion.choices[0].message.role,
            },
            "context": extra_info,
            "session_state": session_state,
        }
