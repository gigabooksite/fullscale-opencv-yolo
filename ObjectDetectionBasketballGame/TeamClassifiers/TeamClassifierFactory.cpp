#include "TeamClassifierFactory.h"
#include "TeamClassifier.h"
#include "DummyTeamClassifier.h"

using namespace TeamClassify;

ITeamClassifier* TeamClassifierFactory::CreateTeamClassifier(const TeamClassifierTypes type)
{
	switch (type)
	{
		case TeamClassifierTypes::TeamClassifier:	return new TeamClassifier();
		case TeamClassifierTypes::Dummy:			return new DummyTeamClassifier();
	}
}