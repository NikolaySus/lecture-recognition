import asyncio
import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


def test_real_stdio_tools_resources_and_error_result(tmp_path):
    async def scenario():
        params = StdioServerParameters(command=sys.executable, args=['-m', 'lecture_recognition.mcp_server',
                                                                    '--jobs-dir', str(tmp_path / 'jobs')])
        async with stdio_client(params) as (reader, writer):
            async with ClientSession(reader, writer) as session:
                await session.initialize()
                tools = (await session.list_tools()).tools
                names = {t.name for t in tools}
                assert {'start_transcription', 'get_segments', 'save_revision', 'get_audio_clip', 'export_transcript'} <= names
                result = await session.call_tool('get_profiles', {})
                assert not result.isError
                assert 'beam' in str(result.structuredContent)
                error = await session.call_tool('get_status', {'job_id': '../wrong'})
                assert error.isError
                resource = await session.read_resource('lecture://workflow')
                assert 'A and B are equal' in resource.contents[0].text
    asyncio.run(scenario())
