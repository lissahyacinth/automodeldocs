from __future__ import annotations
import asyncio
import logging
import pathlib
from typing import TypeAlias, Union, Sequence

from function_discovery import parse_module
from function_discovery.structure import (
    FunctionContainer,
    ClassContainer,
    ModuleContainer,
    ScopeContainer,
)
from dataclasses import dataclass

from automodeldocs.chat.send_message import chat_completion_request
from automodeldocs.config.llm_config import LLMConfig
from automodeldocs.utils import take_items
from automodeldocs.writer import write_description, write_scratch, format_description
from automodeldocs.evaluator.parser import EvaluationResponse
from automodeldocs.evaluator.prompt import Evaluator
from automodeldocs.structures import (
    message_from_system_str,
    message_from_user_str,
    Improvement,
    DescriptionContext,
    Feedback,
)
from automodeldocs.fan_cache import try_load_fan_cache, save_fan_cache

logger = logging.getLogger(__name__)

Container: TypeAlias = Union[FunctionContainer, ClassContainer, ModuleContainer]


@dataclass
class InitialDescription:
    container: Container
    name: str
    description: str
    dependencies: list[
        Union[InitialDescription, ResolvingDescription, ResolvedDescription]
    ]
    feedback: str

    def __repr__(self) -> str:
        return (
            f"InitialDescription(\n\tcontainer: {self.container.__class__},"
            f"\n\tname: {self.name},"
            f"\n\tdescription: {' '.join(take_items(self.description.split(' '), 50))}{'...' if len(self.feedback.split()) > 50 else ''}"
            f"\n\tdependencies: {[str(dependency.__class__) + '(' + dependency.name + ')' for dependency in self.dependencies]}"
            f"\n\tfeedback: {' '.join(take_items(self.feedback.split(' '), 50))}{'...' if len(self.feedback.split()) > 50 else ''}"
            "\n)"
        )

    def __hash__(self) -> int:
        return hash(
            tuple([self.container, self.description]) + tuple(self.dependencies)
        )


@dataclass
class ResolvingDescription:
    container: Container
    name: str
    description: str
    dependencies: list[
        Union[InitialDescription, ResolvingDescription, ResolvedDescription]
    ]
    feedback: str

    def __repr__(self) -> str:
        return (
            f"ResolvingDescription(\n\tcontainer: {self.container.__class__},"
            f"\n\tname: {self.name},"
            f"\n\tdescription: \"{' '.join(take_items(self.description.split(' '), 25))}{'...' if len(self.feedback.split()) > 26 else ''}\"\n"
            f"\n\tdependencies: {[str(dependency.__class__) + '(' + dependency.name + ')' for dependency in self.dependencies]}"
            f"\n\tfeedback: \"{' '.join(take_items(self.feedback.split(' '), 25))}{'...' if len(self.feedback.split()) > 26 else ''}\"\n"
            "\n)"
        )

    def __hash__(self) -> int:
        return hash(
            tuple([self.container, self.description]) + tuple(self.dependencies)
        )


@dataclass
class ResolvedDescription:
    container: Container
    name: str
    description: str
    dependencies: list[
        Union[InitialDescription, ResolvingDescription, ResolvedDescription]
    ]
    feedback: str

    def __repr__(self) -> str:
        return (
            f"ResolvedDescription(\n\tcontainer: {self.container.__class__},"
            f"\n\tname: {self.name},"
            f"\n\tdescription: {' '.join(take_items(self.description.split(' '), 25))}{'...' if len(self.feedback.split()) > 26 else ''}"
            f"\n\tdependencies: {[str(dependency.__class__) + '(' + dependency.name + ')' for dependency in self.dependencies]}"
            f"\n\tfeedback: {' '.join(take_items(self.feedback.split(' '), 25))}{'...' if len(self.feedback.split()) > 26 else ''}"
            "\n)"
        )

    def __hash__(self) -> int:
        return hash(
            tuple([self.container, self.description]) + tuple(self.dependencies)
        )


async def describe_and_try_resolve_node(
    node: InitialDescription | ResolvingDescription, description_context: dict
) -> ResolvedDescription | ResolvingDescription:
    context_node: FunctionContainer | ClassContainer
    description_node: InitialDescription | ResolvingDescription | ResolvedDescription
    current_description, evaluation_response = await fan_and_evaluate(
        node.container.source(),
        node.name,
        node.container.docs(),
        improvement=Improvement(
            feedback=[Feedback(node.description, node.feedback)],
            context=DescriptionContext(
                {
                    context_node.name: description_node.description
                    for (
                        context_node,
                        description_node,
                    ) in description_context.items()
                }
            ),
        ),
    )
    additional_dependencies = [
        node.container.resolve_name(additional_context_item).origin
        for additional_context_item in evaluation_response.additional_context_items
        if node.container.resolve_name(additional_context_item) is not None
        and node.container.resolve_name(additional_context_item).origin
        not in [n.container for n in node.dependencies]
    ]
    class_name = node.name if isinstance(node.container, ClassContainer) else ""
    if len(additional_dependencies) > 0:
        return ResolvingDescription(
            container=node.container,
            name=node.name,
            description=current_description,
            dependencies=node.dependencies
            + list(
                asyncio.gather(
                    *[
                        convert_dependency_to_description(additional_dep, node, class_name)
                        for additional_dep in additional_dependencies
                    ]
                )
            ),
            feedback=evaluation_response.feedback,
        )
    return ResolvedDescription(
        container=node.container,
        name=node.name,
        description=current_description,
        dependencies=node.dependencies,
        feedback=evaluation_response.feedback,
    )


async def resolve_dependencies(
    node: InitialDescription | ResolvingDescription,
    parent_descriptions: set[FunctionContainer | ClassContainer],
) -> InitialDescription | ResolvingDescription:
    for idx in range(len(node.dependencies)):
        dependency = node.dependencies[idx]
        if isinstance(dependency, InitialDescription):
            resolving_node = ResolvingDescription(
                dependency.container,
                dependency.name,
                dependency.description,
                dependency.dependencies,
                feedback=dependency.feedback,
            )
            node.dependencies[idx] = resolving_node
            node.dependencies[idx] = await resolve_description(
                resolving_node, parent_descriptions
            )
        elif isinstance(dependency, ResolvingDescription):
            node.dependencies[idx] = await resolve_description(
                dependency, parent_descriptions
            )
        elif isinstance(dependency, ResolvedDescription):
            pass
    return node


def all_trivial_dependencies_met(
    dependencies: list[InitialDescription | ResolvingDescription | ResolvedDescription],
    parent_descriptions: set,
) -> bool:
    return (
        # No dependencies to meet
        (len(dependencies) == 0)
        # All dependencies are met
        or (all(isinstance(n, ResolvedDescription) for n in dependencies))
        # All dependencies that can be met, are met.
        or (
            all(
                isinstance(dependency, ResolvedDescription)
                for dependency in dependencies
                if dependency.container not in parent_descriptions
            )
        )
    )


async def resolve_description(
    node: InitialDescription | ResolvingDescription,
    parent_descriptions: set[
        FunctionContainer | ClassContainer
    ] = None,  # TODO: This need a better name
) -> ResolvedDescription | ResolvingDescription:
    if parent_descriptions is None:
        parent_descriptions = set()
    description_context: dict[
        FunctionContainer | ClassContainer,
        Union[InitialDescription, ResolvingDescription, ResolvedDescription],
    ] = {}
    print(f"Describing {node.name}")
    parent_descriptions = parent_descriptions | {node.container}
    while not all_trivial_dependencies_met(node.dependencies, parent_descriptions):
        node = await resolve_dependencies(node, parent_descriptions)
    for dependency in node.dependencies:
        if isinstance(dependency, ResolvedDescription):
            description_context[dependency.container] = dependency
    # If there are no dependencies, resolve
    if len(node.dependencies) == 0:
        return await describe_and_try_resolve_node(node, description_context)
    # If the only resolving dependencies of this node are parent functions, resolve this node
    elif all(
        isinstance(dependency, ResolvedDescription)
        or isinstance(dependency, InitialDescription)
        for dependency in node.dependencies
        if dependency.container not in parent_descriptions
    ):
        return await describe_and_try_resolve_node(node, description_context)
    # If any of this node's dependencies have not resolved, this is also still resolving.
    elif any(isinstance(n, ResolvingDescription) for n in node.dependencies):
        return ResolvingDescription(
            node.container,
            node.name,
            node.description,
            node.dependencies,
            feedback=node.feedback,
        )
    # Otherwise all dependencies have resolved, and we can also resolve
    else:
        resolved_node = await describe_and_try_resolve_node(node, description_context)
        return resolved_node


async def create_class_dependency_graph(
    class_info: ClassContainer,
    resolving: dict[
        FunctionContainer | ClassContainer,
        Union[InitialDescription, ResolvingDescription, ResolvedDescription],
    ]
    | None = None,
) -> InitialDescription:
    if resolving is None:
        resolving = {}
    resolving_class_node = InitialDescription(
        container=class_info,
        name=class_info.name,
        description=f"A description of the class {class_info.name}"
        if class_info.docs() is None
        else class_info.docs(),
        dependencies=[],
        feedback="Insufficient information provided about the class.",
    )
    function_dependencies: Sequence[InitialDescription] = await asyncio.gather(
        *[
            create_function_dependency_graph(
                function_info=function,
                class_name=class_info.name,
                resolving=resolving | {class_info: resolving_class_node},
            )
            for function in class_info.functions
        ]
    )
    resolving_class_node.dependencies = list(function_dependencies)
    return resolving_class_node


async def convert_dependency_to_description(
    dependency: FunctionContainer | ClassContainer,
    parent_node: InitialDescription | ResolvingDescription | ResolvedDescription,
    class_name: str | None = None,
    resolving: dict | None = None,
):
    if resolving is None:
        resolving = {}
    if isinstance(dependency, FunctionContainer):
        return await create_function_dependency_graph(
            function_info=dependency,
            class_name=class_name,
            resolving=resolving | {parent_node.container: parent_node},
        )
    elif isinstance(dependency, ClassContainer):
        return await create_class_dependency_graph(
            class_info=dependency,
            resolving=resolving | {parent_node.container: parent_node},
        )


async def create_function_dependency_graph(
    function_info: FunctionContainer,
    class_name: str | None,
    resolving: dict[
        FunctionContainer | ClassContainer,
        Union[InitialDescription, ResolvingDescription, ResolvedDescription],
    ]
    | None = None,
) -> InitialDescription:
    if resolving is None:
        resolving = {}
    # Add in a self-loop rather than recalculating
    if function_info in resolving:
        res = resolving[function_info]
        assert isinstance(res, InitialDescription)
        return res
    if class_name is None:
        function_name = function_info.name
    else:
        function_name = f"{class_name}.{function_info.name}"
    function_docs = function_info.docs()
    function_source = function_info.source()
    initial_description, evaluation_response = await fan_and_evaluate(
        function_source, function_name, function_docs
    )
    dependencies = [
        function_info.resolve_name(additional_context_item).origin
        for additional_context_item in evaluation_response.additional_context_items
        if function_info.resolve_name(additional_context_item) is not None
    ]
    current_node = InitialDescription(
        container=function_info,
        name=function_name,
        description=initial_description,
        feedback=evaluation_response.feedback,
        dependencies=[],
    )
    sub_nodes = list(
        await asyncio.gather(
            *[
                convert_dependency_to_description(
                    dependency,
                    current_node,
                    class_name=class_name,
                    resolving=resolving,
                )
                for dependency in dependencies
                if isinstance(dependency, (FunctionContainer, ClassContainer))
            ]
        )
    )
    current_node.dependencies = list(sub_nodes)
    return current_node


async def fan_and_evaluate(
    function_source: str,
    function_name: str,
    function_docs: str | None,
    improvement: Improvement | None = None,
) -> tuple[str, EvaluationResponse]:
    cache_entry = try_load_fan_cache(
        function_source=function_source,
        function_name=function_name,
        function_docs=function_docs,
        improvement=improvement,
    )
    if cache_entry is not None:
        return cache_entry
    description_strings: list[str] = list(
        await asyncio.gather(
            *[
                write_description(
                    function_source=function_source,
                    function_name=function_name,
                    scratch=(
                        await write_scratch(function_source, function_name, improvement)
                    ),
                    improvement=improvement,
                )
                for _ in range(LLMConfig.from_env().beam_width)
            ]
        )
    )
    if function_docs is not None:
        description_strings += [function_docs]
    evaluation_response = await EvaluationResponse.from_fmt(
        (
            await chat_completion_request(
                [
                    message_from_system_str(
                        Evaluator(description_strings).system_message()
                    ),
                    message_from_user_str(
                        Evaluator(description_strings).user_message()
                    ),
                ]
            )
        ).item[-1]
    )
    best_description = await format_description(
        function_name, description_strings[evaluation_response.documentation_idx]
    )
    save_fan_cache(
        function_source=function_source,
        function_name=function_name,
        function_docs=function_docs,
        improvement=improvement,
        description=best_description,
        evaluation_response=evaluation_response,
    )
    return best_description, evaluation_response


async def fully_describe_item(container: ScopeContainer) -> str:
    if isinstance(container, FunctionContainer):
        base_graph = await create_function_dependency_graph(container, class_name=None)
    elif isinstance(container, ClassContainer):
        base_graph = await create_class_dependency_graph(container)
    else:
        raise ValueError
    resolved_desc = await resolve_description(base_graph)
    breakpoint()
    return resolved_desc.description


if __name__ == "__main__":
    initial_module = parse_module(
        pathlib.Path(r"D:\PaddleOCR\ppocr\modeling\backbones\det_mobilenet_v3.py"),
        module_name="ppocr.modeling.backbones.det_mobilenet_v3",
        starting_path=None,
    )
    print(
        asyncio.run(
            fully_describe_item(initial_module.resolve_name("MobileNetV3").origin)
        )
    )
