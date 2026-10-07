import asyncio
import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from lecture_recognition.jobs import JobStore


def test_real_stdio_tools_resources_and_error_result(tmp_path):
    store = JobStore(tmp_path / 'jobs')
    job, _ = store.create({'audio_info': {'duration': 30}, 'limit_seconds': None})

    async def scenario():
        params = StdioServerParameters(command=sys.executable, args=['-m', 'lecture_recognition.mcp_server',
                                                                    '--jobs-dir', str(tmp_path / 'jobs')])
        async with stdio_client(params) as (reader, writer):
            async with ClientSession(reader, writer) as session:
                await session.initialize()
                tools = (await session.list_tools()).tools
                names = {t.name for t in tools}
                assert {'start_transcription', 'get_segments', 'save_revision', 'get_audio_clip', 'export_transcript', 'save_references', 'get_references'} <= names
                result = await session.call_tool('get_profiles', {})
                assert not result.isError
                assert 'beam' in str(result.structuredContent)
                error = await session.call_tool('get_status', {'job_id': '../wrong'})
                assert error.isError
                saved = await session.call_tool('save_references', {
                    'job_id': job, 'expected_revision': 0,
                    'references': [{'source': 'notes.md', 'original_text': 'Элер', 'text': 'Эйлер',
                                    'window': [1, 5], 'usage': 'both', 'reason': 'Spelling'}]})
                assert not saved.isError
                assert saved.structuredContent['revision'] == 1
                fetched = await session.call_tool('get_references', {'job_id': job})
                assert not fetched.isError
                ref = fetched.structuredContent['references'][0]
                assert ref['window'] == [1.0, 5.0] and ref['confirmed_by'] == 'none'
                assert ref['original_text'] == 'Элер' and ref['text'] == 'Эйлер'
                stale = await session.call_tool('save_references', {
                    'job_id': job, 'expected_revision': 0, 'references': [ref]})
                assert stale.isError
                resource = await session.read_resource('lecture://workflow')
                assert 'A and B are equal' in resource.contents[0].text
        # A fresh MCP process must recover the same materials without the audio file.
        async with stdio_client(params) as (reader, writer):
            async with ClientSession(reader, writer) as session:
                await session.initialize()
                recovered = await session.call_tool('get_references', {'job_id': job})
                assert not recovered.isError
                assert recovered.structuredContent['revision'] == 1
                assert recovered.structuredContent['history'][0]['changes'][0]['original_text'] == 'Элер'
    asyncio.run(scenario())
