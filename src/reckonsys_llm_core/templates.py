from pathlib import Path

from jinja2 import Environment, FileSystemLoader

_env: Environment | None = None


def configure_templates(template_dir: str | Path, **jinja_kwargs) -> None:
    """Configure the Jinja2 environment. Call once at application startup.

    Args:
        template_dir: Directory containing your .j2 / .md template files.
        **jinja_kwargs: Any extra kwargs forwarded to jinja2.Environment
                        (e.g. trim_blocks=True, lstrip_blocks=True).

    Example:
        configure_templates("/path/to/prompts", trim_blocks=True)
    """
    global _env
    jinja_kwargs.setdefault("keep_trailing_newline", True)
    _env = Environment(
        loader=FileSystemLoader(str(template_dir)),
        **jinja_kwargs,
    )


def render_prompt(template_name: str, context: dict | None = None) -> str:
    """Render a template file with the given context.

    Args:
        template_name: Filename relative to the configured template directory.
        context: Variables available inside the template.

    Returns:
        Rendered string ready to be used as a prompt.

    Raises:
        RuntimeError: If called before configure_templates().

    Example:
        render_prompt("system.md.j2", {"user_name": "Alice"})
    """
    if _env is None:
        raise RuntimeError(
            "Templates are not configured. "
            "Call configure_templates(template_dir) before rendering."
        )
    return _env.get_template(template_name).render(context or {})
