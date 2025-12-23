from pip._vendor.resolvelib.providers import Preference
from pip._vendor.resolvelib.resolvers import RequirementInformation

PreferenceInformation = RequirementInformation[Requirement, Candidate]

_ProviderBase = AbstractProvider[Requirement, Candidate, str]
