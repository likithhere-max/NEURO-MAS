#!/usr/bin/env python3
"""
Build the disease knowledge base from free Orphadata products.

Requires: Python 3.10+, requests, pandas, lxml
Install: pip install requests pandas lxml

The Orphanet API repository documents these free products:
product1  = disease nomenclature + terminology mappings
product3  = clinical classifications
product4  = HPO phenotypes
product6  = genes
product7  = linearisation
product9_prev = epidemiology
product9_ages = natural history

The metadata endpoints below are documented by the public Orphadata project.
If a URL is changed by Orphadata, update PRODUCTS accordingly.
"""
import io, json, zipfile, re
from pathlib import Path
import requests
import pandas as pd
from lxml import etree

OUT=Path("generated")
OUT.mkdir(exist_ok=True)

PRODUCTS = {
 "product1":"http://www.orphadata.org/cgi-bin/free_product1_cross_xml.json",
 "product3":"http://www.orphadata.org/cgi-bin/free_product3_class.json",
 "product4":"http://www.orphadata.org/cgi-bin/free_product4_hpo.json",
 "product6":"http://www.orphadata.org/cgi-bin/free_product6_genes.json",
 "product7":"http://www.orphadata.org/cgi-bin/free_product7_linear.json",
 "product9_prev":"http://www.orphadata.org/cgi-bin/free_product9_prev.json",
 "product9_ages":"http://www.orphadata.org/cgi-bin/free_product9_ages.json",
}

def get_json(url):
    r=requests.get(url,timeout=60)
    r.raise_for_status()
    return r.json()

def find_url(obj):
    if isinstance(obj,dict):
        for k,v in obj.items():
            if isinstance(v,str) and (v.startswith("http://") or v.startswith("https://")):
                if ".xml" in v.lower() or ".json" in v.lower():
                    return v
            u=find_url(v)
            if u:return u
    elif isinstance(obj,list):
        for x in obj:
            u=find_url(x)
            if u:return u
    return None

def download_products():
    urls={}
    for name,meta in PRODUCTS.items():
        try:
            meta_json=get_json(meta)
            u=find_url(meta_json)
            if not u:
                print("No downloadable URL found for",name)
                continue
            urls[name]=u
            print("Downloading",name,u)
            r=requests.get(u,timeout=180)
            r.raise_for_status()
            (OUT/f"{name}.xml").write_bytes(r.content)
        except Exception as e:
            print("FAILED",name,repr(e))
    return urls

def xml_text(el, xpath):
    x=el.find(xpath)
    return (x.text or "").strip() if x is not None else ""

def parse_product1(path):
    tree=etree.parse(str(path))
    rows=[]
    for d in tree.xpath("//*[local-name()='Disorder']"):
        orpha=xml_text(d,"OrphaCode") or xml_text(d,"OrphaNumber")
        name=xml_text(d,"Name")
        if not name: continue
        synonyms=[x.text.strip() for x in d.xpath(".//*[local-name()='SynonymName']") if x.text]
        rows.append({
            "orpha_code":orpha,
            "preferred_name":name,
            "synonyms":" | ".join(dict.fromkeys(synonyms)),
        })
    return pd.DataFrame(rows).drop_duplicates(subset=["orpha_code"])

def parse_hpo(path):
    tree=etree.parse(str(path)); rows=[]
    for d in tree.xpath("//*[local-name()='Disorder']"):
        orpha=xml_text(d,"OrphaCode") or xml_text(d,"OrphaNumber")
        for a in d.xpath(".//*[local-name()='HPODisorderAssociation']"):
            hpo=""
            for n in a.xpath(".//*[local-name()='HPOId']"):
                if n.text: hpo=n.text.strip(); break
            term=""
            for n in a.xpath(".//*[local-name()='HPOName']"):
                if n.text: term=n.text.strip(); break
            freq=""
            for n in a.xpath(".//*[local-name()='HPOFrequency']"):
                if n.text: freq=n.text.strip(); break
            if hpo:
                rows.append({"orpha_code":orpha,"hpo_id":hpo,"hpo_term":term,"frequency":freq})
    return pd.DataFrame(rows).drop_duplicates()

def parse_genes(path):
    tree=etree.parse(str(path)); rows=[]
    for d in tree.xpath("//*[local-name()='Disorder']"):
        orpha=xml_text(d,"OrphaCode") or xml_text(d,"OrphaNumber")
        for a in d.xpath(".//*[local-name()='ExternalReference']"):
            source=xml_text(a,"Source")
            ref=xml_text(a,"Reference")
            if source and ref and source.upper() in {"HGNC","OMIM"}:
                rows.append({"orpha_code":orpha,"database":source,"gene_or_reference":ref})
    return pd.DataFrame(rows).drop_duplicates()

def main():
    download_products()
    p1=OUT/"product1.xml"
    if p1.exists():
        df=parse_product1(p1)
        df.to_csv(OUT/"orphanet_diseases.csv",index=False)
        print("Disease entities:",len(df))
    p4=OUT/"product4.xml"
    if p4.exists():
        df=parse_hpo(p4)
        df.to_csv(OUT/"disease_hpo_phenotypes.csv",index=False)
        print("Phenotype links:",len(df))
    p6=OUT/"product6.xml"
    if p6.exists():
        df=parse_genes(p6)
        df.to_csv(OUT/"disease_gene_links.csv",index=False)
        print("Gene links:",len(df))
    print("Done. Add treatment evidence from current clinical guidelines into treatment_evidence.csv.")

if __name__=="__main__":
    main()
