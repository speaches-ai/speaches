<!-- https://mkdocstrings.github.io/python/usage/configuration/general/ -->

!!! note

    Also checkout [customizing HuggingFace behaviour](https://huggingface.co/docs/huggingface_hub/package_reference/environment_variables#environment-variables)

::: speaches.config.Config
    options:
        show_bases: true
        show_if_no_docstring: true
        show_labels: false
        separate_signature: true
        show_signature_annotations: true
        signature_crossrefs: true
        summary: false
        source: true
        members_order: source
        filters:
            - "!model_config"

::: speaches.config.WhisperConfig

<!-- TODO: nested model `whisper`  -->
<!-- TODO: Insert new lines for multi-line docstrings  -->
