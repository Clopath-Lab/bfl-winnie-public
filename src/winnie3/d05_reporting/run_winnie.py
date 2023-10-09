import sys
from pathlib import Path
from kedro.runner import SequentialRunner
from kedro.framework.session import KedroSession
from winnie3.d07_pipelines.deploy import create_deploy_pipeline
from kedro.framework.session.session import _activate_session


def run_winnie(case_id: int, extra_words: str = [], run_winnie: bool = True):
    """Estimated and returns a set of candidate responses to a new question(s)
    :param case_id: id of the case being queried
    :param extra_words: extra words to be searched for in str format separated by comma e.g. 'land rights,divorce'
    :param run_winnie: boolean indicating whether case data should be used or just search with the extra words
    """
    _clear_hook_manager()
    package_name = "winnie3"
    _remove_cached_modules(package_name)
    pipe = create_deploy_pipeline(
        case_id=case_id,
        run_winnie=run_winnie,
        extra_words=extra_words,
        search_importance=1.0,
    )
    runner = SequentialRunner()
    path = Path.cwd()
    session = KedroSession.create(package_name, path)
    _activate_session(session, force=True)
    print("Loading the context from %s", str(path))
    context = session.load_context()
    result = runner.run(pipe, context.catalog)
    return True


def _clear_hook_manager():
    from kedro.framework.hooks import get_hook_manager

    hook_manager = get_hook_manager()
    name_plugin_pairs = hook_manager.list_name_plugin()
    for name, plugin in name_plugin_pairs:
        hook_manager.unregister(name=name, plugin=plugin)  # pragma: no cover


def _remove_cached_modules(package_name):
    to_remove = [mod for mod in sys.modules if mod.startswith(package_name)]
    # `del` is used instead of `reload()` because: If the new version of a module does not
    # define a name that was defined by the old version, the old definition remains.
    for module in to_remove:
        del sys.modules[module]  # pragma: no cover
