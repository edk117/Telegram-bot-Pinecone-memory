"""Клиент для работы с векторами в индексе Pinecone.

Перед использованием добавьте ``PINECONE_API_KEY`` в файл ``.env`` или
передайте ключ в конструктор. Сам индекс должен быть создан в Pinecone заранее.
"""

import os
from typing import Any, Mapping, Optional, Sequence

from dotenv import load_dotenv
from pinecone import Pinecone


class PineconeVectorClient:
    """Предоставляет операции записи, чтения, поиска и удаления векторов.

    Args:
        index_name: Имя существующего индекса Pinecone.
        api_key: Ключ Pinecone. Если не указан, берётся из ``PINECONE_API_KEY``.
        namespace: Пространство имён по умолчанию. Пустая строка означает
            пространство имён по умолчанию Pinecone.
    """

    def __init__(
        self,
        index_name: str,
        api_key: Optional[str] = None,
        namespace: str = "",
    ) -> None:
        """Подключается к существующему индексу Pinecone.

        Args:
            index_name: Имя индекса для подключения.
            api_key: API-ключ Pinecone или ``None`` для чтения из окружения.
            namespace: Пространство имён для операций по умолчанию.

        Raises:
            ValueError: Если имя индекса или API-ключ не заданы.
        """
        load_dotenv()
        resolved_api_key = api_key or os.getenv("PINECONE_API_KEY")
        if not index_name:
            raise ValueError("Необходимо указать имя индекса Pinecone.")
        if not resolved_api_key:
            raise ValueError(
                "Не найден API-ключ Pinecone. Передайте api_key или задайте "
                "PINECONE_API_KEY."
            )

        self.namespace = namespace
        self._pinecone = Pinecone(api_key=resolved_api_key)
        self._index = self._pinecone.Index(name=index_name)

    def upsert_vectors(
        self,
        vectors: Sequence[Mapping[str, Any]],
        namespace: Optional[str] = None,
    ) -> Any:
        """Записывает новые векторы или обновляет существующие.

        Args:
            vectors: Последовательность словарей Pinecone с ключами ``id`` и
                ``values``; необязательный ключ ``metadata`` содержит метаданные.
            namespace: Пространство имён. При ``None`` используется значение из
                конструктора.

        Returns:
            Ответ Pinecone с числом обработанных векторов (``upserted_count``).
        """
        return self._index.upsert(
            vectors=list(vectors), namespace=self._resolve_namespace(namespace)
        )

    def fetch_vectors(
        self,
        vector_ids: Sequence[str],
        namespace: Optional[str] = None,
    ) -> Any:
        """Читает векторы по их идентификаторам.

        Args:
            vector_ids: Идентификаторы запрашиваемых векторов.
            namespace: Пространство имён. При ``None`` используется значение из
                конструктора.

        Returns:
            Ответ Pinecone, содержащий найденные векторы и их метаданные.
        """
        return self._index.fetch(
            ids=list(vector_ids), namespace=self._resolve_namespace(namespace)
        )

    def search_vectors(
        self,
        query_vector: Sequence[float],
        top_k: int = 5,
        metadata_filter: Optional[Mapping[str, Any]] = None,
        namespace: Optional[str] = None,
        include_values: bool = False,
        include_metadata: bool = True,
    ) -> Any:
        """Ищет ближайшие к запросу векторы.

        Args:
            query_vector: Вектор той же размерности, что и индекс.
            top_k: Максимальное число наиболее похожих результатов; больше нуля.
            metadata_filter: Необязательный фильтр Pinecone по метаданным.
            namespace: Пространство имён. При ``None`` используется значение из
                конструктора.
            include_values: Включать ли значения найденных векторов в ответ.
            include_metadata: Включать ли метаданные найденных векторов в ответ.

        Returns:
            Ответ Pinecone со списком совпадений и их оценками сходства.

        Raises:
            ValueError: Если ``top_k`` не является положительным числом.
        """
        if top_k < 1:
            raise ValueError("top_k должен быть больше нуля.")

        return self._index.query(
            vector=list(query_vector),
            top_k=top_k,
            filter=dict(metadata_filter) if metadata_filter is not None else None,
            namespace=self._resolve_namespace(namespace),
            include_values=include_values,
            include_metadata=include_metadata,
        )

    def delete_vectors(
        self,
        vector_ids: Sequence[str],
        namespace: Optional[str] = None,
    ) -> None:
        """Удаляет векторы с указанными идентификаторами.

        Args:
            vector_ids: Идентификаторы векторов, которые нужно удалить.
            namespace: Пространство имён. При ``None`` используется значение из
                конструктора.

        Returns:
            ``None``. Pinecone выполняет операцию удаления асинхронно.
        """
        self._index.delete(
            ids=list(vector_ids), namespace=self._resolve_namespace(namespace)
        )

    def _resolve_namespace(self, namespace: Optional[str]) -> str:
        """Возвращает явно переданное или установленное по умолчанию пространство."""
        return self.namespace if namespace is None else namespace
