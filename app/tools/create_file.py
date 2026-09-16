"""Generate a real downloadable artifact in the authenticated conversation."""
import json

from app.tools.registry import ToolDefinition


def create_file(filename: str, content: str) -> str:
    from app.services.artifact_context import artifact_owner
    from backend.services.artifact_service import create_file as store_file
    from app.agents.events import emit

    user_id, conversation_id = artifact_owner()
    artifact = store_file(user_id, conversation_id, filename, content)
    emit('file_operation', 'file_created', '文件已生成', filename,
         status='completed', artifact=artifact)
    return json.dumps(artifact, ensure_ascii=False)


TOOL = ToolDefinition(
    name='create_file',
    description='生成可预览、可下载的真实文件：Word(docx)、Excel(xlsx)、PDF，以及 md/txt/csv/json/html。doc/xls 文件名会转为 docx/xlsx。Word/PDF 的 content 使用 Markdown，支持标题、列表、表格。Excel 的 content 使用 JSON 字符串，例：{"sheets":[{"name":"销售","rows":[["产品","金额"],["A",100]]}]}；也接受 CSV。最多20个工作表，每行100列，总计10万单元格。Excel 字符串按纯文本写入，不执行公式；请直接填入计算后的数值。文件最大5 MiB。保存成功后提示用户在右侧产物中预览或下载，不要编造链接。',
    fn=create_file,
    arg_schema={'filename': ('string', '带扩展名的文件名，不含路径', True),
                'content': ('string', 'docx/pdf: Markdown 正文；xlsx: 工作表 JSON 字符串或 CSV；其他格式: 完整正文。不要包裹代码围栏', True)},
    metadata={'public': True, 'tags': ['file', 'export', 'report'],
              'scenarios': ['文件', '下载', '报告', '导出', '文档', '表格', 'word', 'excel', 'pdf', 'docx', 'xlsx', 'csv', 'html', 'markdown']},
)
