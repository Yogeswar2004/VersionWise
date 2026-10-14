# Pydantic V1 Documentation Sample

This is a synthetic test fixture for VersionWise Phase 3 ingestion unit tests.

## Custom Validators

Pydantic V1 uses the `@validator` decorator for custom validation logic on model fields.

```python
from pydantic import BaseModel, validator

class UserModel(BaseModel):
    name: str

    @validator('name')
    def name_must_contain_space(cls, v):
        if ' ' not in v:
            raise ValueError('must contain a space')
        return v
```

Validators in V1 are run by default in order of declaration unless configured otherwise.

## Settings Management

Settings in V1 are defined by subclassing `BaseSettings`.

```python
from pydantic import BaseSettings

class Settings(BaseSettings):
    app_name: str = "TestApp"
```

