import pytest
from xdiyo_analytics.data.scoreline_decisions import candidates, parse_answers, request_for


def test_country_shortlist_abstention_and_never_accept_out_of_set():
    queue=[dict(league='Premier_League',season='2009/2010',source='example',source_team_key='x',
                names='QPR Queens Park Rangers')]
    clubs=[dict(country=x,slug='queens-park-rangers',team_id=i,names=['Queens Park Rangers'],seasons=['2011/2012'])
           for x,i in [('England',1),('France',2)]]
    cases=candidates(queue,clubs,25)
    assert len(cases)==1 and [x['team_id'] for x in cases[0]['options']]==[1]
    req,allow=request_for(cases)
    assert req['model']=='gpt-6-luna'
    assert req['questions'][0]['type']=='choice'
    answer=dict(type='choice',name='team_0',choice='team 1',confidence=.91)
    suggestions=parse_answers(dict(answers=[answer]),cases,allow)
    assert suggestions[0]['status']=='needs_review'
    assert suggestions[0]['proposed_sofascore_team_id']==1
    assert parse_answers(dict(answers=[dict(answer,choice='no_confident_match')]),cases,allow)[0]['status']=='abstained'
    assert parse_answers(dict(answers=[dict(name='team_0',type='refusal')]),cases,allow)[0]['status']=='refusal'
    with pytest.raises(ValueError,match='shortlist'):
        parse_answers(dict(answers=[dict(answer,choice='team 2')]),cases,allow)
    with pytest.raises(ValueError,match='Incomplete'):
        parse_answers(dict(answers=[]),cases,allow)
