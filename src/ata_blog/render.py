from pathlib import Path
from jinja2 import Environment, FileSystemLoader, select_autoescape

def make_env(template_dir: Path):
    return Environment(
        loader=FileSystemLoader(str(template_dir)),
        autoescape=select_autoescape(enabled_extensions=('html', 'xml'))
    )

def render_markdown(template_dir: Path, template_name: str, context: dict) -> str:
    env = make_env(template_dir)
    tpl = env.get_template(template_name)
    return tpl.render(**context)
