# Pydantic V2 Documentation Sample

This is a synthetic test fixture for VersionWise Phase 3 ingestion unit tests.

## Field Validators

Pydantic V2 replaces `@validator` with the `@field_validator` decorator and requires the `@classmethod` decorator.

```python
from pydantic import BaseModel, field_validator

class UserModel(BaseModel):
    name: str

    @field_validator('name')
    @classmethod
    def name_must_contain_space(cls, v: str) -> str:
        if ' ' not in v:
            raise ValueError('must contain a space')
        return v
```

Mode settings such as `mode='before'` or `mode='after'` can be customized on `@field_validator`.

## Settings Migration

In V2, `BaseSettings` has been moved to a separate package named `pydantic-settings`.

```python
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    app_name: str = "TestApp"
    model_config = SettingsConfigDict(env_file=".env")
```

