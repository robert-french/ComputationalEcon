"""
Build script for marimo notebooks.

This script uses notebooks/ as the single source of truth.

Each marimo notebook in notebooks/ is exported twice:
    1. As an editable notebook in _site/notebooks/
    2. As a read-only app in _site/apps/

The script also generates an index.html file that lists both versions.
"""

# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "jinja2==3.1.3",
#     "fire==0.7.0",
#     "loguru==0.7.0"
# ]
# ///

import shutil
import subprocess
from typing import List, Union
from pathlib import Path

import jinja2
import fire

from loguru import logger


def _export_html_wasm(
    notebook_path: Path,
    output_file: Path,
    as_app: bool = False,
) -> bool:
    """Export a single marimo notebook to HTML/WebAssembly format."""

    cmd: List[str] = ["uvx", "marimo", "export", "html-wasm", "--sandbox"]

    if as_app:
        logger.info(f"Exporting {notebook_path} to {output_file} as app")
        cmd.extend(["--mode", "run", "--no-show-code"])
    else:
        logger.info(f"Exporting {notebook_path} to {output_file} as editable notebook")
        cmd.extend(["--mode", "edit"])

    try:
        output_file.parent.mkdir(parents=True, exist_ok=True)

        cmd.extend([str(notebook_path), "-o", str(output_file)])

        logger.debug(f"Running command: {cmd}")
        subprocess.run(cmd, capture_output=True, text=True, check=True)

        logger.info(f"Successfully exported {notebook_path}")
        return True

    except subprocess.CalledProcessError as e:
        logger.error(f"Error exporting {notebook_path}:")
        logger.error(f"Command output: {e.stderr}")
        return False

    except Exception as e:
        logger.error(f"Unexpected error exporting {notebook_path}: {e}")
        return False


def _generate_index(
    output_dir: Path,
    template_file: Path,
    notebooks_data: List[dict] | None = None,
    apps_data: List[dict] | None = None,
) -> None:
    """Generate an index.html file that lists all exported notebooks and apps."""

    logger.info("Generating index.html")

    index_path: Path = output_dir / "index.html"
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        template_dir = template_file.parent
        template_name = template_file.name

        env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(template_dir),
            autoescape=jinja2.select_autoescape(["html", "xml"]),
        )

        template = env.get_template(template_name)
        rendered_html = template.render(notebooks=notebooks_data, apps=apps_data)

        with open(index_path, "w") as f:
            f.write(rendered_html)

        logger.info(f"Successfully generated index.html at {index_path}")

    except IOError as e:
        logger.error(f"Error generating index.html: {e}")

    except jinja2.exceptions.TemplateError as e:
        logger.error(f"Error rendering template: {e}")


def _export_from_notebooks(
    source_folder: Path,
    output_dir: Path,
    as_app: bool = False,
) -> List[dict]:
    """Export all marimo notebooks from notebooks/ either as apps or editable notebooks.

    Source files always live in notebooks/.

    Editable notebooks are written to:
        _site/notebooks/<name>.html

    Apps are written to:
        _site/apps/<name>.html
    """

    if not source_folder.exists():
        logger.warning(f"Directory not found: {source_folder}")
        return []

    notebooks = list(source_folder.rglob("*.py"))
    logger.debug(f"Found {len(notebooks)} Python files in {source_folder}")

    if not notebooks:
        logger.warning(f"No notebooks found in {source_folder}!")
        return []

    notebook_data = []

    output_subfolder = "apps" if as_app else "notebooks"

    for nb in notebooks:
        relative_path = nb.relative_to(source_folder).with_suffix(".html")
        html_path = Path(output_subfolder) / relative_path
        output_file = output_dir / html_path

        success = _export_html_wasm(
            notebook_path=nb,
            output_file=output_file,
            as_app=as_app,
        )

        if success:
            notebook_data.append(
                {
                    "display_name": nb.stem.replace("_", " ").title(),
                    "html_path": str(html_path),
                }
            )

    logger.info(
        f"Successfully exported {len(notebook_data)} out of {len(notebooks)} files "
        f"from {source_folder} as {'apps' if as_app else 'editable notebooks'}"
    )

    return notebook_data


def main(
    output_dir: Union[str, Path] = "_site",
    template: Union[str, Path] = "templates/tailwind.html.j2",
) -> None:
    """Main function to export marimo notebooks."""

    logger.info("Starting marimo build process")

    output_dir = Path(output_dir)
    logger.info(f"Output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    template_file = Path(template)
    logger.info(f"Using template file: {template_file}")

    source_folder = Path("notebooks")

    # Export each source notebook once as an editable notebook.
    notebooks_data = _export_from_notebooks(
        source_folder=source_folder,
        output_dir=output_dir,
        as_app=False,
    )

    # Export each source notebook again as a clean app.
    apps_data = _export_from_notebooks(
        source_folder=source_folder,
        output_dir=output_dir,
        as_app=True,
    )

    if not notebooks_data and not apps_data:
        logger.warning("No notebooks found!")
        return

    _generate_index(
        output_dir=output_dir,
        notebooks_data=notebooks_data,
        apps_data=apps_data,
        template_file=template_file,
    )

    # Copy any static assets sitting next to the template (e.g. the LMU shield
    # watermark) into the site root so the rendered index.html can reference
    # them with relative paths.
    static_dir = template_file.parent / "static"
    if static_dir.exists():
        for asset in static_dir.iterdir():
            if asset.is_file():
                shutil.copy2(asset, output_dir / asset.name)
                logger.info(f"Copied static asset {asset.name} to {output_dir}")

    logger.info(f"Build completed successfully. Output directory: {output_dir}")


if __name__ == "__main__":
    fire.Fire(main)
