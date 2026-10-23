from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from app.core.supabase_client import get_authed_client
from app.schemas.chat import ChatMessageIn
from app.services.auth import get_current_user
from app.services.chat_assistant import answer_patient_question
from app.services.repository import build_case_context

router = APIRouter(prefix="/patients", tags=["chat"])


@router.get("/{patient_id}/chat/history")
def get_chat_history(patient_id: str, user: dict = Depends(get_current_user)):
    client = get_authed_client(user["_access_token"])
    patient_result = client.table("patients").select("id").eq("id", patient_id).execute()
    if not patient_result.data:
        raise HTTPException(404, "patient not found")
    result = (
        client.table("chat_messages")
        .select("*")
        .eq("patient_id", patient_id)
        .order("created_at")
        .execute()
    )
    return result.data


@router.post("/{patient_id}/chat")
async def send_chat_message(
    patient_id: str,
    payload: ChatMessageIn,
    user: dict = Depends(get_current_user),
):
    """
    Context-aware AI Assistant (section 10/11 of the brief). Builds a
    compact case-context summary from the patient's most recent workflow
    (app.services.repository.build_case_context) plus recent chat history,
    and answers strictly from that context via
    app.services.chat_assistant.answer_patient_question. Persists both the
    user's message and the assistant's reply.
    """
    if not payload.message or not payload.message.strip():
        raise HTTPException(422, "message is required")

    client = get_authed_client(user["_access_token"])
    patient_result = client.table("patients").select("*").eq("id", patient_id).execute()
    if not patient_result.data:
        raise HTTPException(404, "patient not found")
    patient = patient_result.data[0]

    history_result = (
        client.table("chat_messages")
        .select("*")
        .eq("patient_id", patient_id)
        .order("created_at")
        .execute()
    )
    history = history_result.data or []

    context = build_case_context(client, patient_id, patient=patient)

    result = await answer_patient_question(
        question=payload.message, context=context, history=history
    )

    rows_to_insert = [
        {"patient_id": patient_id, "role": "user", "content": payload.message},
        {"patient_id": patient_id, "role": "assistant", "content": result["answer"]},
    ]
    insert_result = client.table("chat_messages").insert(rows_to_insert).execute()
    saved_assistant_message = (
        insert_result.data[1] if insert_result.data and len(insert_result.data) > 1 else None
    )

    return {
        "answer": result["answer"],
        "used_context": result["used_context"],
        "warnings": result["warnings"],
        "is_emergency": bool(result.get("is_emergency", False)),
        "message": saved_assistant_message,
    }
