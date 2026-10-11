"""Canonical v0.7: typed core, inspection and schedule/material profiles."""
from datetime import datetime
from typing import Annotated, Literal
from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator

Id = Annotated[str, Field(min_length=1)]
Basis = Literal['evidence', 'inferred', 'hypothesis', 'mixed']
Visibility = Literal['ai_visible', 'evaluation_only']

class Model(BaseModel):
    model_config = ConfigDict(extra='forbid')

class Attribution(Model):
    basis: Basis
    evidence_ids: list[Id] = Field(default_factory=list)
    assumption_ids: list[Id] = Field(default_factory=list)

class Provenance(Attribution):
    # JSON pointers into the record. Whole-record basis remains authoritative.
    field_attributions: dict[str, Attribution] = Field(default_factory=dict)
    generator_rule_id: Id | None = None

class Record(Model):
    schema_version: Literal['0.7'] = '0.7'
    created_at: AwareDatetime  # serialization/record creation, NOT knowledge time
    source_type: Literal['synthetic'] = 'synthetic'
    provenance: Provenance

class TemporalRecord(Record):
    valid_from: AwareDatetime
    known_at: AwareDatetime
    valid_to: AwareDatetime | None = None
    valid_to_known_at: AwareDatetime | None = None

    @model_validator(mode='after')
    def interval(self):
        if (self.valid_to is None) != (self.valid_to_known_at is None):
            raise ValueError('valid_to and valid_to_known_at must be supplied together')
        if self.valid_to is not None and self.valid_to <= self.valid_from:
            raise ValueError('valid_to must be > valid_from (half-open nonempty interval)')
        if self.valid_to_known_at is not None and self.valid_to_known_at < self.known_at:
            raise ValueError('valid_to_known_at must be >= known_at')
        return self

class EntityBase(TemporalRecord):
    entity_id: Id
    project_id: Id

class ProjectAttributes(Model):
    name: str
    project_category: Literal['electrical_installation'] = 'electrical_installation'
    initial_status: Literal['planned'] = 'planned'

class ActivityAttributes(Model):
    name: str
    activity_type: Literal['installation', 'inspection', 'correction']
    initial_status: Literal['planned'] = 'planned'
    inspection_type: Literal['visual_check'] | None = None
    inspection_of_id: Id | None = None
    inspection_attempt: Annotated[int, Field(ge=1)] | None = None
    previous_inspection_id: Id | None = None

    @model_validator(mode='after')
    def inspection_fields(self):
        if self.activity_type == 'inspection':
            if self.inspection_type is None or self.inspection_of_id is None or self.inspection_attempt is None:
                raise ValueError('inspection requires type, target and attempt')
            if (self.inspection_attempt == 1) != (self.previous_inspection_id is None):
                raise ValueError('only repeated inspections require previous_inspection_id')
        elif any(x is not None for x in (self.inspection_type, self.inspection_of_id,
                                         self.inspection_attempt, self.previous_inspection_id)):
            raise ValueError('non-inspection Activity cannot contain inspection attributes')
        return self

class ActorAttributes(Model):
    actor_kind: Literal['person', 'company', 'team', 'organization', 'unknown'] = 'person'
    role: Literal['worker', 'inspector', 'site_supervisor', 'approver']

class IssueAttributes(Model):
    issue_kind: Literal['quality', 'inspection_finding', 'schedule', 'safety', 'other']
    summary: str  # private human description, not an AI field
    initial_status: Literal['open'] = 'open'

class LocationAttributes(Model):
    name: str
    location_type: Literal['work_area', 'site', 'other']

class ArtifactAttributes(Model):
    name: str
    artifact_type: Literal['photo', 'report', 'drawing', 'document', 'other']

class Project(EntityBase):
    entity_type: Literal['project'] = 'project'
    attributes: ProjectAttributes

class Activity(EntityBase):
    entity_type: Literal['activity'] = 'activity'
    attributes: ActivityAttributes

class Actor(EntityBase):
    entity_type: Literal['actor'] = 'actor'
    attributes: ActorAttributes

class Issue(EntityBase):
    entity_type: Literal['issue'] = 'issue'
    attributes: IssueAttributes

class Location(EntityBase):
    entity_type: Literal['location'] = 'location'
    attributes: LocationAttributes

class Artifact(EntityBase):
    entity_type: Literal['artifact'] = 'artifact'
    attributes: ArtifactAttributes

class MaterialAttributes(Model):
    name: str
    material_category: Literal['cable_support']

class Material(EntityBase):
    entity_type: Literal['material'] = 'material'
    attributes: MaterialAttributes

Entity = Annotated[Project | Activity | Actor | Issue | Location | Artifact | Material, Field(discriminator='entity_type')]
RelationType = Literal['PART_OF', 'PRECEDES', 'DEPENDS_ON', 'ASSIGNED_TO', 'LOCATED_AT',
                       'REFERENCES', 'OBSERVES', 'DETECTS', 'CORRECTS', 'VERIFIES', 'APPROVES', 'SUPERSEDES', 'REQUIRES_MATERIAL']

class Relationship(TemporalRecord):
    relationship_id: Id
    source_id: Id
    target_id: Id
    relation_type: RelationType

EventType = Literal['PROJECT_CREATED', 'PROJECT_STARTED', 'ACTIVITY_CREATED', 'ACTIVITY_STARTED',
    'ACTIVITY_COMPLETED', 'INSPECTION_PERFORMED', 'INSPECTION_FAILED', 'CORRECTION_REQUESTED',
    'CORRECTION_STARTED', 'CORRECTION_COMPLETED', 'REINSPECTION_PERFORMED', 'INSPECTION_PASSED',
    'ISSUE_RAISED', 'ISSUE_RESOLVED', 'ISSUE_CLOSED', 'ISSUE_REOPENED',
    'MATERIAL_UNAVAILABLE', 'MATERIAL_AVAILABLE', 'BLOCKING_STARTED', 'BLOCKING_CLEARED',
    'SCHEDULE_REVISED', 'RISK_ASSESSED', 'SCHEDULE_ISSUE_RESOLVED']

class EventPayload(Model):
    delivery_plan_id: Id | None = None
    blocking_condition_id: Id | None = None
    schedule_revision_id: Id | None = None
    risk_assessment_id: Id | None = None
    evidence_event_ids: list[Id] = Field(default_factory=list)
    result: Literal['fail', 'pass'] | None = None
    issue_ids: list[Id] = Field(default_factory=list)

class Event(Record):
    event_id: Id
    event_type: EventType
    sequence: Annotated[int, Field(ge=1)]
    occurred_at: AwareDatetime
    recorded_at: AwareDatetime
    project_id: Id
    subject_id: Id
    actor_ids: list[Id]
    payload: EventPayload

    @model_validator(mode='after')
    def time(self):
        if self.recorded_at < self.occurred_at:
            raise ValueError('recorded_at must be >= occurred_at')
        return self

class EventLink(Record):
    event_link_id: Id
    project_id: Id
    source_event_id: Id
    target_event_id: Id
    relation_type: Literal['CAUSES', 'TRIGGERS', 'PRECEDES', 'ASSOCIATED_WITH']
    basis: Literal['observed', 'inferred', 'synthetic_ground_truth']
    confidence: Annotated[float, Field(ge=0, le=1)]
    known_at: AwareDatetime
    visibility: Visibility

    @model_validator(mode='after')
    def private_truth(self):
        if self.basis == 'synthetic_ground_truth' and self.visibility != 'evaluation_only':
            raise ValueError('synthetic_ground_truth EventLink must be evaluation_only')
        return self

class PrivateText(Model):
    text: str
    visibility: Literal['evaluation_only'] = 'evaluation_only'

class ObservationBase(Record):
    observation_id: Id
    project_id: Id
    subject_ids: Annotated[list[Id], Field(min_length=1)]
    source_event_ids: Annotated[list[Id], Field(min_length=1)]
    observed_at: AwareDatetime
    recorded_at: AwareDatetime
    visibility: Visibility = 'ai_visible'
    private_text: PrivateText | None = None

    @model_validator(mode='after')
    def time(self):
        if self.recorded_at < self.observed_at:
            raise ValueError('recorded_at must be >= observed_at')
        return self

class InspectionContent(Model):
    inspection_id: Id
    check_item: Literal['cable_identification']
    result: Literal['fail', 'pass']
    finding: Literal['label_missing', 'label_present']
    issue_ids: Annotated[list[Id], Field(min_length=1)]

class ProgressContent(Model):
    activity_id: Id
    reported_progress: Annotated[float, Field(ge=0, le=1)]

class MessageContent(Model):
    issue_ids: Annotated[list[Id], Field(min_length=1)]
    message_kind: Literal['correction_request', 'site_note']
    statement: Literal['attach_identification_label', 'location_confirmed_label_missing']

    @model_validator(mode='after')
    def statement_kind(self):
        expected = {'correction_request': 'attach_identification_label',
                    'site_note': 'location_confirmed_label_missing'}
        if self.statement != expected[self.message_kind]:
            raise ValueError('message kind and statement disagree')
        return self

class PhotoContent(Model):
    activity_id: Id
    issue_ids: Annotated[list[Id], Field(min_length=1)]
    depicts: Literal['installed_identification_label_visible']
    # No invented URI; optional artifact ref is checked if supplied.
    artifact_id: Id | None = None

class ReportContent(Model):
    activity_id: Id
    report_kind: Literal['completion']
    reported_progress: Literal[1.0]

class InspectionObservation(ObservationBase):
    observation_type: Literal['inspection_record'] = 'inspection_record'
    content: InspectionContent

class ProgressObservation(ObservationBase):
    observation_type: Literal['progress_record'] = 'progress_record'
    content: ProgressContent

class MessageObservation(ObservationBase):
    observation_type: Literal['message'] = 'message'
    content: MessageContent

class PhotoObservation(ObservationBase):
    observation_type: Literal['photo'] = 'photo'
    content: PhotoContent

class ReportObservation(ObservationBase):
    observation_type: Literal['report'] = 'report'
    content: ReportContent

Observation = Annotated[InspectionObservation | ProgressObservation | MessageObservation |
                        PhotoObservation | ReportObservation, Field(discriminator='observation_type')]

class PhysicalRootCause(Model):
    status: Literal['unknown', 'identified']
    description: str | None = None

    @model_validator(mode='after')
    def description_status(self):
        if (self.status == 'identified') != bool(self.description):
            raise ValueError('identified physical cause requires description; unknown has none')
        return self

class InspectionOutcome(Model):
    outcome_type: Literal['inspection'] = 'inspection'
    event_id: Id
    result: Literal['pass', 'fail']
    issue_status: Literal['open', 'pending_verification', 'closed']

class ActivityScheduleOutcome(Model):
    activity_id: Id
    baseline_revision_id: Id
    revised_revision_id: Id | None = None
    actual_start_event_id: Id
    actual_end_event_id: Id
    start_deviation_minutes: int
    end_deviation_minutes: int
    revised_start_deviation_minutes: int | None = None
    revised_end_deviation_minutes: int | None = None
    realized_delay: bool
    unit: Literal['minute'] = 'minute'

class ScheduleOutcome(Model):
    outcome_type: Literal['schedule'] = 'schedule'
    activities: Annotated[list[ActivityScheduleOutcome], Field(min_length=1)]
    affected_downstream_activity_ids: list[Id]
    # No numeric causal attribution: baseline difference is not a causal estimate.
    causal_contribution: Literal['not_estimated'] = 'not_estimated'

# Commercial records are immutable business facts, not accounting entries.
class Money(Model):
    model_config = ConfigDict(extra='forbid', frozen=True)
    amount: Annotated[int, Field(strict=True, ge=0)]
    currency: Literal['JPY']
    tax_basis: Literal['excluded', 'included', 'unspecified']

class CommercialRecord(Record):
    model_config = ConfigDict(extra='forbid', frozen=True)
    project_id: Id
    valid_from: AwareDatetime
    known_at: AwareDatetime
    activity_ids: tuple[Id, ...] = ()
    private_text: PrivateText | None = None

    @model_validator(mode='after')
    def commercial_time(self):
        if self.known_at < self.valid_from:
            raise ValueError('commercial known_at must not precede valid_from')
        return self

class Estimate(CommercialRecord):
    revision_id: Id
    estimate_id: Id
    version: Annotated[int, Field(strict=True, ge=1)]
    previous_revision_id: Id | None = None
    total_amount: Money

class ExecutionBudget(CommercialRecord):
    revision_id: Id
    budget_id: Id
    version: Annotated[int, Field(strict=True, ge=1)]
    previous_revision_id: Id | None = None
    estimate_revision_id: Id
    planned_cost_amount: Money
    change_id: Id | None = None

class Order(CommercialRecord):
    order_id: Id
    amount: Money
    budget_revision_id: Id
    counterparty_role: Literal['supplier_or_subcontractor']
    original_order_id: Id | None = None
    change_id: Id | None = None

class CommercialChange(CommercialRecord):
    change_id: Id
    category: Literal['additional_work']
    base_budget_revision_id: Id
    base_order_ids: Annotated[tuple[Id, ...], Field(min_length=1)]

class AcceptanceAllocation(Model):
    model_config = ConfigDict(extra='forbid', frozen=True)
    order_id: Id
    accepted_amount: Money

class AcceptanceBase(CommercialRecord):
    acceptance_id: Id
    order_allocations: Annotated[tuple[AcceptanceAllocation, ...], Field(min_length=1)]

class FullAcceptance(AcceptanceBase):
    acceptance_kind: Literal['completion_full']
    completion_event_ids: Annotated[tuple[Id, ...], Field(min_length=1)]

class PartialAcceptance(AcceptanceBase):
    acceptance_kind: Literal['progress_partial']
    progress_observation_ids: Annotated[tuple[Id, ...], Field(min_length=1)]

Acceptance = Annotated[FullAcceptance | PartialAcceptance, Field(discriminator='acceptance_kind')]

class OrderAllocation(Model):
    model_config = ConfigDict(extra='forbid', frozen=True)
    order_id: Id
    amount: Money

class Invoice(CommercialRecord):
    invoice_id: Id
    amount: Money
    order_allocations: Annotated[tuple[OrderAllocation, ...], Field(min_length=1)]
    acceptance_ids: Annotated[tuple[Id, ...], Field(min_length=1)]

class InvoiceAssessment(CommercialRecord):
    assessment_id: Id
    invoice_id: Id
    decision: Literal['approved']
    assessed_amount: Money

class InvoiceAllocation(Model):
    model_config = ConfigDict(extra='forbid', frozen=True)
    invoice_id: Id
    assessment_id: Id
    amount: Money

class Payment(CommercialRecord):
    payment_id: Id
    amount: Money
    invoice_allocations: Annotated[tuple[InvoiceAllocation, ...], Field(min_length=1)]

class CommercialNotice(Record):
    observation_id: Id
    project_id: Id
    observation_type: Literal['commercial_record_notice'] = 'commercial_record_notice'
    target_kind: Literal['estimates', 'execution_budgets', 'orders', 'commercial_changes',
                         'acceptances', 'invoices', 'invoice_assessments', 'payments']
    target_id: Id
    observed_at: AwareDatetime
    recorded_at: AwareDatetime
    visibility: Visibility = 'ai_visible'
    private_text: PrivateText | None = None

    @model_validator(mode='after')
    def notice_time(self):
        if self.recorded_at < self.observed_at:
            raise ValueError('notice recorded before observed')
        return self

class CommercialBalances(Model):
    model_config = ConfigDict(extra='forbid', frozen=True)
    ordered_amount: Money
    accepted_amount: Money
    remaining_to_accept: Money
    invoiced_amount: Money
    remaining_to_invoice: Money  # commitment minus invoices, NOT an accrued payable
    accepted_uninvoiced_amount: Money
    assessed_amount: Money
    unassessed_amount: Money  # invoice minus assessed, includes assessment reductions
    paid_amount: Money
    remaining_to_pay: Money  # approved amount minus payment
    invoice_unpaid_amount: Money

class CommercialSnapshot(Model):
    as_of: AwareDatetime
    known: Literal[True] = True
    balances: CommercialBalances | None  # None when no Order is known

class FutureCommercialRecord(Model):
    table: Literal['estimates', 'execution_budgets', 'orders', 'commercial_changes',
                   'acceptances', 'invoices', 'invoice_assessments', 'payments']
    record_id: Id

class CommercialOutcome(Model):
    outcome_type: Literal['commercial'] = 'commercial'
    state_at_evaluation: CommercialSnapshot
    future_records: tuple[FutureCommercialRecord, ...]
    final_state: CommercialSnapshot
    final_settled: bool

Outcome = Annotated[InspectionOutcome | ScheduleOutcome | CommercialOutcome, Field(discriminator='outcome_type')]

class GroundTruth(Record):
    ground_truth_id: Id
    project_id: Id
    evaluation_as_of: AwareDatetime
    root_event_id: Id
    origin_kind: Literal['workflow_origin'] = 'workflow_origin'
    physical_root_cause: PhysicalRootCause
    event_link_ids: list[Id]
    affected_entity_ids: Annotated[list[Id], Field(min_length=1)]
    future_event_ids: list[Id]  # complete set within project, after evaluation_as_of
    final_outcome: Outcome
    visibility: Literal['evaluation_only'] = 'evaluation_only'

class Evidence(Model):
    url: str
    retrieved_at: AwareDatetime
    supports: str

class Assumption(Model):
    description: str
    version: str = '1'

class InspectionMetadata(Record):
    profile: Literal['inspection-v0.7']
    scenario: Literal['inspection_failure']
    seed: None = None
    generator_version: Literal['0.7.0'] = '0.7.0'
    evidence_register: dict[str, Evidence]
    assumption_register: dict[str, Assumption]

class ScheduleMetadata(Record):
    profile: Literal['schedule-material-v0.7']
    scenario: Literal['schedule_material_delay', 'schedule_material_early_warning', 'schedule_multi_material_branching']
    seed: None = None
    generator_version: Literal['0.7.0'] = '0.7.0'
    evidence_register: dict[str, Evidence]
    assumption_register: dict[str, Assumption]

class CommercialMetadata(Record):
    profile: Literal['commercial-v0.7']
    scenario: Literal['commercial_additional_work', 'commercial_progress_billing']
    seed: None = None
    generator_version: Literal['0.7.0'] = '0.7.0'
    evidence_register: dict[str, Evidence]
    assumption_register: dict[str, Assumption]

Metadata = Annotated[InspectionMetadata | ScheduleMetadata | CommercialMetadata, Field(discriminator='profile')]

class PlanWindow(Model):
    model_config = ConfigDict(extra='forbid', frozen=True)
    activity_id: Id
    planned_start: AwareDatetime
    planned_end: AwareDatetime

    @model_validator(mode='after')
    def window(self):
        if self.planned_end <= self.planned_start:
            raise ValueError('planned_end must be after planned_start')
        return self

class ScheduleRevision(Record):
    model_config = ConfigDict(extra='forbid', frozen=True)
    schedule_revision_id: Id
    project_id: Id
    version: Annotated[int, Field(ge=1)]
    previous_revision_id: Id | None = None
    valid_from: AwareDatetime
    known_at: AwareDatetime
    assignments: Annotated[tuple[PlanWindow, ...], Field(min_length=1)]

class DeliveryPlan(Record):
    model_config = ConfigDict(extra='forbid', frozen=True)
    delivery_plan_id: Id
    project_id: Id
    material_id: Id
    expected_at: AwareDatetime
    valid_from: AwareDatetime
    known_at: AwareDatetime

class BlockingCondition(Record):
    blocking_condition_id: Id
    project_id: Id
    activity_id: Id
    reason: Literal['required_material_unavailable']
    material_id: Id
    delivery_plan_id: Id
    valid_from: AwareDatetime
    known_at: AwareDatetime

class RiskAssessment(Record):
    risk_assessment_id: Id
    previous_assessment_id: Id | None = None
    project_id: Id
    target_activity_ids: Annotated[list[Id], Field(min_length=1)]
    assessed_at: AwareDatetime
    known_at: AwareDatetime
    risk_kind: Literal['schedule_delay']
    level: Literal['low', 'medium', 'high']
    assessor_kind: Literal['human', 'system', 'synthetic']
    baseline_revision_id: Id
    supporting_event_ids: list[Id] = Field(default_factory=list)
    supporting_observation_ids: list[Id] = Field(default_factory=list)

    @model_validator(mode='after')
    def time(self):
        if self.known_at < self.assessed_at:
            raise ValueError('risk known_at precedes assessed_at')
        if not (self.supporting_event_ids or self.supporting_observation_ids):
            raise ValueError('risk requires Event or Observation evidence')
        return self

class MaterialContent(Model):
    material_id: Id
    delivery_plan_id: Id
    availability: Literal['unavailable', 'available']

class ScheduleRevisionContent(Model):
    schedule_revision_id: Id
    activity_ids: Annotated[list[Id], Field(min_length=1)]

class ScheduleRiskContent(Model):
    risk_assessment_id: Id
    activity_ids: Annotated[list[Id], Field(min_length=1)]
    risk_level: Literal['low', 'medium', 'high']

class MaterialObservation(ObservationBase):
    observation_type: Literal['material_availability'] = 'material_availability'
    content: MaterialContent

class ScheduleObservation(ObservationBase):
    observation_type: Literal['schedule_revision'] = 'schedule_revision'
    content: ScheduleRevisionContent

class RiskObservation(ObservationBase):
    observation_type: Literal['schedule_risk'] = 'schedule_risk'
    content: ScheduleRiskContent

class DeliveryWarningContent(Model):
    warning_kind: Literal['delivery_delay_risk']
    severity: Literal['low', 'medium', 'high']
    delivery_plan_id: Id

class DeliveryWarningObservation(ObservationBase):
    observation_type: Literal['delivery_warning'] = 'delivery_warning'
    source_event_ids: list[Id] = Field(default_factory=list)
    content: DeliveryWarningContent

class DeliveryUpdateContent(Model):
    delivery_plan_id: Id
    status: Literal['dispatched', 'dispatch_delayed']
    estimated_arrival_at: AwareDatetime

class DeliveryUpdateObservation(ObservationBase):
    observation_type: Literal['delivery_update'] = 'delivery_update'
    source_event_ids: list[Id] = Field(default_factory=list)
    content: DeliveryUpdateContent

    @model_validator(mode='after')
    def prospective_eta(self):
        if self.content.estimated_arrival_at < self.observed_at:
            raise ValueError('estimated_arrival_at precedes observed_at')
        return self

Observation = Annotated[InspectionObservation | ProgressObservation | MessageObservation |
    PhotoObservation | ReportObservation | MaterialObservation | ScheduleObservation | RiskObservation | DeliveryWarningObservation | DeliveryUpdateObservation,
    Field(discriminator='observation_type')]

class Dataset(Model):
    entities: list[Entity]
    relationships: list[Relationship]
    events: list[Event]
    event_links: list[EventLink]
    observations: list[Observation]
    ground_truth: list[GroundTruth]
    metadata: Metadata
    schedule_revisions: list[ScheduleRevision]
    delivery_plans: list[DeliveryPlan]
    blocking_conditions: list[BlockingCondition]
    risk_assessments: list[RiskAssessment]

    estimates: list[Estimate]
    execution_budgets: list[ExecutionBudget]
    orders: list[Order]
    commercial_changes: list[CommercialChange]
    acceptances: list[Acceptance]
    invoices: list[Invoice]
    invoice_assessments: list[InvoiceAssessment]
    payments: list[Payment]
    commercial_observations: list[CommercialNotice]

    def world_state(self, as_of: datetime):
        from .views import world_state
        return world_state(self, as_of)

    def build_ai_view(self, as_of: datetime):
        from .views import build_ai_view
        return build_ai_view(self, as_of)
