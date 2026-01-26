RELATION_DESCRIPTION = {
    "org:alternate_names": "an organization's alternate names",
    "org:city_of_headquarters": "an organization's city of headquarters",
    "org:country_of_headquarters": "an organization's country of headquarters",
    "org:dissolved": "an organization's date of dissolution",
    "org:founded": "an organization's date of founding",
    "org:founded_by": "an organization's founder",
    "org:member_of": "an organization's membership of another entity",
    "org:members": "an organization's members",
    "org:number_of_employees/members": "an organization's number of employees or members",
    "org:parents": "an organization's parents",
    "org:political/religious_affiliation": "an organization's political or religious affiliation",
    "org:shareholders": "an organization's shareholders",
    "org:stateorprovince_of_headquarters": "an organization's state or province of headquarters",
    "org:subsidiaries": "an organization's subsidiaries",
    "org:top_members/employees": "an organization's top members or employees",
    "org:website": "an organization's website",
    "per:age": "a person's age",
    "per:alternate_names": "a person's alternate names",
    "per:cause_of_death": "a person's cause of death",
    "per:charges": "a person's criminal charges",
    "per:children": "a person's children",
    "per:cities_of_residence": "a person's cities of residence",
    "per:city_of_birth": "a person's city of birth",
    "per:city_of_death": "a person's city of death",
    "per:countries_of_residence": "a person's countries of residence",
    "per:country_of_birth": "a person's country of birth",
    "per:country_of_death": "a person's country of death",
    "per:date_of_birth": "a person's date of birth",
    "per:date_of_death": "a person's date of death",
    "per:employee_of": "a person's employer",
    "per:origin": "a person's city or country of origin",
    "per:other_family": "a person's other family",
    "per:parents": "a person's parents",
    "per:religion": "a person's religion",
    "per:schools_attended": "schools attended by a person",
    "per:siblings": "a person's siblings",
    "per:spouse": "a person's spouse",
    "per:stateorprovince_of_birth": "a person's state or province of birth",
    "per:stateorprovince_of_death": "a person's state or province of death",
    "per:stateorprovinces_of_residence": "a person's state or province of residence",
    "per:title": "a person's title",

    ### RE-tacred
    # "org:city_of_branch": "an organization's city of branch",
    # "org:stateorprovince_of_branch": "an organization's state or province of branch",
    # "org:country_of_branch": "an organization's country of branch",
    # "per:identity": "a person's identity",
}

def get_relation_description(relation, dt = "fs_tacred"):
    if dt == "fs_tacred":
        assert relation in RELATION_DESCRIPTION.keys()

        return RELATION_DESCRIPTION[relation]
    else:
        return ""