#include "TeamClassifierFactory.h"
#include "TeamClassifier.h"
#include "DummyTeamClassifier.h"

using namespace TeamClassify;

ITeamClassifier* TeamClassifierFactory::CreateTeamClassifier(const TeamClassifierTypes type)
{
	ITeamClassifier* ptr = nullptr;
	switch (type)
	{
		case TeamClassifierTypes::TeamClassifier:	
			ptr = new TeamClassifier();
			break;
		case TeamClassifierTypes::Dummy:
			//break; intentional fall-through
		default:
			ptr = new DummyTeamClassifier();
			break;
	}

	return ptr;
}