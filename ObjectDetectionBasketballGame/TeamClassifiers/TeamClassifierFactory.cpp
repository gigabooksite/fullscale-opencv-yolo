#include "TeamClassifierFactory.h"
#include "TeamClassifier.h"
#include "DummyTeamClassifier.h"

using namespace TeamClassify;

ITeamClassifier* TeamClassifierFactory::CreateTeamClassifier(const std::string& type)
{
	if (type == "dummy")
	{
		return new DummyTeamClassifier();
	}
	else if (type == "teamclassifier")
	{
		return new TeamClassifier();
	}
	else
	{
		return nullptr;
	}
}