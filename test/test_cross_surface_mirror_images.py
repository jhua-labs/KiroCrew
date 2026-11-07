"""The dashboard -> channel mirror sends pictures, never this host's paths.

The reply leg uploads a local image through the transport's proactive verb under
the channel renderers' own gates; the user-message echo names a dashboard
attachment instead of printing its server path.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from chat_test_helpers import _make_state

# A real SEL under tmp_path, so the legs' audit writes stay out of the user's log.
from test_cross_surface_mirror import audit  # noqa: F401

from kiro_crew.dashboard import chat_runner
from kiro_crew.dashboard.chat_runner import (
    _deliver_cross_surface_reply,
    _deliver_cross_surface_user_message,
    _name_local_attachments,
)
from kiro_crew.feishu.transport import FeishuTransport
from kiro_crew.messaging.link import ChannelLink
from kiro_crew.messaging.outbound_files import ExtractLimits

_PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32


def _transport(*, uploads: bool = True) -> SimpleNamespace:
    return SimpleNamespace(
        channel_type="feishu",
        capabilities=SimpleNamespace(
            supports_proactive_send=True,
            files_outbound=uploads,
            max_message_chars=4000,
            max_message_bytes=0,
        ),
        outbound_image_limits=lambda conversation_id, thread_id=None: ExtractLimits(
            max_files=10, max_file_bytes=1024 * 1024
        ),
        send_message=AsyncMock(return_value=""),
        send_outbound_image=AsyncMock(return_value=True),
        may_send_to=lambda conversation_id, thread_id=None, principal="": True,
    )


def _state(tmp_path, monkeypatch, transport, *, restricted: bool = False):
    work = tmp_path / "work"
    work.mkdir()
    (work / "chart.png").write_bytes(_PNG)
    state = _make_state(tmp_path)
    state.register_channel_transport(transport)
    state.sessions.get_mirror_link = MagicMock(
        return_value=ChannelLink("feishu", channel_id="ou_owner", thread_id=None)
    )
    state.sessions.get_provider = MagicMock(
        return_value=SimpleNamespace(cwd=str(work), work_scratch_dir=None)
    )
    monkeypatch.setattr(chat_runner, "uploads_restricted", AsyncMock(return_value=restricted))
    return state, work


@pytest.mark.asyncio
@pytest.mark.usefixtures("audit")
async def test_a_reply_image_under_the_cwd_is_uploaded_not_printed(tmp_path, monkeypatch):
    tp = _transport()
    state, work = _state(tmp_path, monkeypatch, tp)
    path = work / "chart.png"

    await _deliver_cross_surface_reply(state, "dashboard:c1", f"Here it is\n\n![chart]({path})")

    (sent,) = [c.args[1] for c in tp.send_message.await_args_list]
    assert str(path) not in sent and "Here it is" in sent
    (_, image), kwargs = tp.send_outbound_image.await_args
    assert image.data == _PNG and kwargs == {"thread_id": None}


@pytest.mark.asyncio
@pytest.mark.usefixtures("audit")
async def test_a_restricted_session_keeps_the_text_and_sends_no_file(tmp_path, monkeypatch):
    tp = _transport()
    state, work = _state(tmp_path, monkeypatch, tp, restricted=True)

    await _deliver_cross_surface_reply(state, "dashboard:c1", f"![chart]({work / 'chart.png'})")

    tp.send_outbound_image.assert_not_awaited()
    assert str(work / "chart.png") in tp.send_message.await_args.args[1]


@pytest.mark.asyncio
@pytest.mark.usefixtures("audit")
async def test_a_refused_reference_keeps_its_path_and_names_the_refusal(tmp_path, monkeypatch):
    tp = _transport()
    state, _ = _state(tmp_path, monkeypatch, tp)
    outside = tmp_path / "elsewhere.png"
    outside.write_bytes(_PNG)

    await _deliver_cross_surface_reply(state, "dashboard:c1", f"see ![x]({outside})")

    tp.send_outbound_image.assert_not_awaited()
    answer, note = [c.args[1] for c in tp.send_message.await_args_list]
    assert str(outside) in answer and note.startswith("⚠")


@pytest.mark.asyncio
@pytest.mark.usefixtures("audit")
async def test_the_echo_names_dashboard_attachments_instead_of_their_paths(tmp_path, monkeypatch):
    tp = _transport()
    state, _ = _state(tmp_path, monkeypatch, tp)
    uploads = "/home/u/.kiro/crew/uploads"
    message = (
        f"look\n![image]({uploads}/a1_Screenshot.png)\n"
        f"and [attached_file 1] {uploads}/b2_deck.pptx"
    )

    await _deliver_cross_surface_user_message(state, "dashboard:c1", message)

    echo = tp.send_message.await_args.args[1]
    assert uploads not in echo
    assert "🖼 a1_Screenshot.png" in echo and "📎 b2_deck.pptx" in echo
    tp.send_outbound_image.assert_not_awaited()


def test_a_message_without_attachments_is_unchanged() -> None:
    assert _name_local_attachments("plain `![x](/a.png)` text") == "plain `![x](/a.png)` text"


@pytest.mark.asyncio
async def test_feishu_uploads_to_an_allow_listed_dm_only() -> None:
    client = SimpleNamespace(send_new_image=AsyncMock(return_value=True))
    transport = FeishuTransport(
        client,  # type: ignore[arg-type]
        allowed_open_ids=["ou_owner"],
        allow_group=True,
        allowed_group_ids=["oc_group"],
    )
    image = SimpleNamespace(data=_PNG)

    assert transport.outbound_image_limits("oc_group") is None
    assert transport.outbound_image_limits("ou_owner") is not None
    assert await transport.send_outbound_image("oc_group", image) is False  # type: ignore[arg-type]
    client.send_new_image.assert_not_awaited()
    assert await transport.send_outbound_image("ou_owner", image) is True  # type: ignore[arg-type]
    client.send_new_image.assert_awaited_once_with("open_id", "ou_owner", _PNG)
