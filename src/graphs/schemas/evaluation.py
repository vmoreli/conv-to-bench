from pydantic import BaseModel, Field
from typing import List

class CheckItem(BaseModel):
    reasoning: str = Field(
        ...,
        description="The reasoning behind evaluation of this criteria. Explain briefly why you considered it true or false."
    )
    check: bool = Field(
        ...,
        description="Boolean that indicates if the respective criteria/requirement was satisfied or not."
    )

class ChecksSchema(BaseModel):
    items: List[CheckItem] = Field(
        ...,
        description="List of items indicating whether each checklist item is satisfied, in the same order as provided."
    )