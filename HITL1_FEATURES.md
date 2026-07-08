# HITL-1 Enhanced Features

This document describes the three new capabilities added to HITL-1 (the skill decomposition review step) that allow users to manually override the skill tree structure without triggering an LLM re-run.

## Overview

HITL-1 is where the human reviews the skill tree decomposed for each layer before the pipeline advances. Previously, users could only edit node names and descriptions. These three new features provide deeper structural control:

1. **Type Toggle**: Convert a node between atomic and composite without re-decomposition
2. **Subskill CRUD**: Add, remove, and edit sub-skills on composite nodes
3. **Exec-Type Editor**: Change the execution type of atomic nodes at review time

All changes are "manual overrides only"—no LLM re-runs—consistent with the existing `/edit` and `/edit_children` endpoints.

---

## Feature 1: Type Toggle (Atomic ↔ Composite)

### UI Location
In the **Layer View** tab, each node card shows a **clickable type badge** (blue for atomic, blue for composite).

### Behavior
- Click the type badge to toggle the node's type
- A confirmation dialog appears with the implications
- **Atomic → Composite**:
  - Clears the node's `exec_type`, `implementation`, and `instruction`
  - Sets `composition_type` to `SEQUENTIAL` (default)
  - Adds 2 placeholder sub-skills if the node currently has fewer than 2
- **Composite → Atomic**:
  - Clears all children and `composition_type`
  - Sets `exec_type` to `LLM_PROMPT` (default)
  - Returns to atomicity; the pipeline considers this resolved for that node

### Implementation
- Backend: `POST /convert_node/{node_id}` endpoint in `src/ui/server.py`
- Frontend: `convertNode(nodeId, currentType, btn)` function
- Payload: `{target_type: "atomic" | "composite", exec_type?: str, composition_type?: str}`

---

## Feature 2: Subskill CRUD (Add / Remove / Edit)

### UI Location
In the **Layer View** tab, composite nodes show a **children-edit-list** section with:
- Inline editable sub-skill names (contenteditable span)
- Inline editable sub-skill descriptions (textarea)
- **× Remove button** on each row (right side)
- **+ Add sub-skill button** below the list

### Behavior

**Editing existing sub-skills**:
- Click on a sub-skill name or description to edit inline
- Save button persists all changes to the composite node

**Adding a new sub-skill**:
- Click "+ Add sub-skill"
- A new row appears with "New Sub-skill" placeholder name
- Edit the name and description
- Save button syncs to the backend (no `id` indicates a new child)

**Removing a sub-skill**:
- Click the × button on any row
- If removing would leave fewer than 2 children, a toast error appears and removal is blocked
- Otherwise, the row is removed from the DOM
- Save button deletes the child from the tree

**Minimum enforced**: Composite nodes must always have ≥ 2 sub-skills. Attempting to save with fewer triggers a validation error.

### Implementation
- Backend: Extended `PUT /edit_children/{node_id}` in `src/ui/server.py`
- Frontend functions:
  - `addChildRow(nodeId)` — appends a new `.child-edit-row` with no `data-child-id`
  - `removeChildRow(btn, nodeId)` — removes a row with >= 2 check
  - Updated `saveComposite(nodeId, btn)` — handles new and deleted children
- Payload: `{description?: str, children: [{id?: str, name: str, description: str}], composition_type?: str}`
  - Entries **with `id`** = update existing child
  - Entries **without `id`** = create new child
  - Omitted children = delete them

---

## Feature 3: Exec-Type Editor (Atomic Nodes Only)

### UI Location
In the **Layer View** tab, atomic nodes show a **dropdown select** next to the type badge with options:
- `llm prompt` (LLM_PROMPT)
- `external api` (EXTERNAL_API)
- `deterministic code` (DETERMINISTIC_CODE)

Composite nodes do not show this dropdown (they use `composition_type` instead).

### Behavior
- Clicking the dropdown changes the selected `exec_type`
- The change is persisted when you click "Save edits" on the node
- No validation: any atomic node can be any `exec_type`

### Implementation
- Backend: Extended `POST /edit/{node_id}` in `src/ui/server.py`
- Frontend: Updated `saveNode(nodeId, btn)` to read the select value and send `exec_type` in the payload
- Payload: `{name?: str, description?: str, exec_type?: str}`

---

## Technical Details

### Backend Changes (`src/ui/server.py`)

#### 1. Extended `EditPayload` (line 726)
```python
class EditPayload(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    exec_type: Optional[str] = None  # NEW
```

#### 2. Updated `edit_node` endpoint (line 732)
```python
@app.post("/edit/{node_id}")
async def edit_node(node_id: str, payload: EditPayload) -> dict:
    # ...existing code...
    if payload.exec_type is not None:
        try:
            node.exec_type = ExecType(payload.exec_type)
        except ValueError:
            raise HTTPException(status_code=422, detail=f"Invalid exec_type: {payload.exec_type}")
```

#### 3. Extended `EditChildrenPayload` (line 761)
```python
class EditChildrenPayload(BaseModel):
    description: Optional[str] = None
    children: list[dict]  # {id?: str, name: str, description: str}
    composition_type: Optional[str] = None  # NEW
```

#### 4. Refactored `edit_children` endpoint (line 761)
Now supports:
- **Updating** existing children (by id)
- **Creating** new children (entries without id)
- **Deleting** children (omit their id from the submission)
- **Changing** composition type

#### 5. New `ConvertNodePayload` and `/convert_node/{node_id}` endpoint (line 839)
```python
class ConvertNodePayload(BaseModel):
    target_type: str  # "atomic" | "composite"
    exec_type: Optional[str] = None
    composition_type: Optional[str] = None

@app.post("/convert_node/{node_id}")
async def convert_node(node_id: str, payload: ConvertNodePayload) -> dict:
    # Atomic → Composite: add placeholders, clear exec_type/impl/instruction
    # Composite → Atomic: clear children/composition_type, set exec_type
```

### Frontend Changes (`src/ui/templates/dashboard.html`)

#### 1. New CSS classes (line 529-625)
- `.type-toggle-btn` — clickable type badge
- `select.exec-type-select` — exec-type dropdown
- `.btn-add-child` — add sub-skill button
- `.btn-remove-child` — remove sub-skill button
- `.child-row-wrapper` / `.child-row-content` — layout helpers

#### 2. Updated node card template (line 1560-1610)
- Type badge → toggle button with `onclick="convertNode(...)"`
- Exec badge → select dropdown (for atomic nodes only)
- Add/remove buttons on child rows
- Updated child-edit-row structure with flex wrapper

#### 3. Updated JS functions
- `saveNode(nodeId, btn)` — now captures and sends `exec_type`
- `saveComposite(nodeId, btn)` — handles new children (no `id`), removed children, `composition_type`
- **New** `convertNode(nodeId, currentType, btn)` — POST to `/convert_node`, reload on success
- **New** `addChildRow(nodeId)` — insert new row into children-edit-list
- **New** `removeChildRow(btn, nodeId)` — remove row with >= 2 validation

---

## Validation Rules

| Rule | Enforced By | Behavior |
|---|---|---|
| Composite nodes must have ≥ 2 children | Backend + Frontend | Toast error on save/add |
| Atomic nodes must have `exec_type` set | (None, but convention) | Never appears as undefined in UI |
| Composite nodes have no `exec_type` | Backend (cleared on conversion) | Null value in SkillNode |
| Child nodes get correct `depth` and `parent_id` | Backend | Auto-set when creating via `/edit_children` |
| Only valid ExecType/CompositionType values accepted | Backend (Enum validation) | HTTP 422 on invalid value |

---

## User Workflow Example

1. **HITL-1 review flow** starts, Layer 1 nodes displayed
2. **User sees** an atomic node "Validate Input" with type=atomic, exec_type=llm_prompt
3. **User realizes** it should be composite (multiple sub-steps needed)
4. **User clicks** the type badge, confirms the conversion
5. **Page reloads**, node is now composite with 2 placeholder children
6. **User edits** the placeholder names:
   - "Check Format" (exists)
   - "Check Constraints" (exists)
7. **User adds** a third sub-skill: "+ Add sub-skill" → edits to "Check Completeness"
8. **User saves** the composite node
9. **Backend** persists: the 3 children, composition_type=SEQUENTIAL
10. **User** clicks "Approve & advance"
11. **Pipeline** moves to implementation review with the updated structure

---

## Testing

Run the included test file to verify all features:
```bash
python test_hitl1_features.py
```

Manual testing:
1. Run `python main.py --web` and start a pipeline
2. Reach `awaiting_review` status on Layer 1
3. Try each feature:
   - Toggle a node's type
   - Add/remove/edit sub-skills
   - Change an atomic node's exec-type
   - Attempt to save with < 2 children (should fail)
4. Click "Approve & advance" to confirm the structure persists
5. Implementation review should show the updated structure

---

## Backwards Compatibility

- Existing nodes without `exec_type` are safe (it's Optional)
- UI shows exec-type dropdown **only for atomic nodes** (won't confuse composite users)
- The minimum 2-child constraint matches the existing `/edit_children` validation
- No changes to the tree structure or persistence—purely UI/endpoint additions

---

## Future Enhancements

- Allow user to set `composition_type` (SEQUENTIAL, PARALLEL, LOOP, LLM_COORDINATOR) from UI
- Per-child exec_type hints (gray out non-sensical options)
- Undo/redo for type conversions
- Drag-to-reorder children on composite nodes
