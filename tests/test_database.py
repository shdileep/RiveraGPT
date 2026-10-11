import uuid
import pytest
from database import (
    init_db,
    create_or_update_conversation,
    list_conversations,
    toggle_pin_conversation,
    rename_conversation,
    delete_conversation,
    save_chat_message,
    get_chat_history,
    save_memory,
    search_memory
)


@pytest.fixture(autouse=True)
def setup_database():
    init_db()


def test_conversation_lifecycle():
    thread_id = f"test_thread_{uuid.uuid4().hex[:8]}"

    # Create conversation
    create_or_update_conversation(thread_id, "Hello from unit test")
    convs = list_conversations()
    target = next((c for c in convs if c.thread_id == thread_id), None)
    assert target is not None
    assert target.title.startswith("Hello from unit test")
    assert target.pinned is False

    # Pin conversation
    new_status = toggle_pin_conversation(thread_id)
    assert new_status is True

    # Rename conversation
    success = rename_conversation(thread_id, "Renamed Test Title")
    assert success is True
    convs_after_rename = list_conversations()
    target_renamed = next((c for c in convs_after_rename if c.thread_id == thread_id), None)
    assert target_renamed.title == "Renamed Test Title"

    # Messages
    save_chat_message(thread_id, "user", "Test prompt")
    save_chat_message(thread_id, "assistant", "Test answer")
    history = get_chat_history(thread_id)
    assert len(history) == 2
    assert history[0].role == "user"
    assert history[0].content == "Test prompt"
    assert history[1].role == "assistant"
    assert history[1].content == "Test answer"

    # Delete conversation
    del_success = delete_conversation(thread_id)
    assert del_success is True
    assert len(get_chat_history(thread_id)) == 0


def test_memory_save_and_search():
    thread_id = f"test_mem_{uuid.uuid4().hex[:8]}"
    save_memory(thread_id, "Favorite programming language is Python")
    res = search_memory(thread_id, "Python")
    assert "Favorite programming language is Python" in res


def test_conversation_document_persistence():
    from database import save_conversation_document, get_conversation_documents
    thread_id = f"test_doc_{uuid.uuid4().hex[:8]}"
    save_conversation_document(thread_id, "report.pdf", ".pdf", "Revenue in Q3 was $10M and growth was 25%.")
    docs = get_conversation_documents(thread_id)
    assert len(docs) == 1
    assert docs[0].filename == "report.pdf"
    assert "Revenue in Q3 was $10M" in docs[0].content

    # Delete conversation cleans up documents
    delete_conversation(thread_id)
    docs_after_delete = get_conversation_documents(thread_id)
    assert len(docs_after_delete) == 0

