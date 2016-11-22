#!/usr/bin/env python3
import pandas
import matplotlib.pyplot


def main():
    # init
    decade_map = {0: '0-9',
                  1: '10-19',
                  2: '20-29',
                  3: '30-39',
                  4: '40-49',
                  5: '50-59',
                  6: '60-69',
                  7: '70-79'}
    pclass_map = {'*': 0,
                  '1st': 1,
                  '2nd': 2,
                  '3rd': 3}

    d = pandas.read_csv('data.csv')
    num_passengers = len(d)
    num_survivors = d.loc[:, 'Survived'].sum()
    d['Decade'] = d.loc[:, 'Age'].dropna().apply(lambda x: decade_map[x // 10])
    matplotlib.pyplot.style.use('ggplot')

    # 1.
    print("Titanic passengers:", num_passengers, '', sep='\n')

    # 2.
    print("Overall survivors:", str(num_survivors / num_passengers * 100) + " %\n", sep='\n')

    # 3.
    sex = d.loc[:, ['Survived', 'SexCode']].groupby('SexCode').sum()
    print("Male survivors:", str(sex.loc[0, 'Survived'] / num_passengers * 100) + " %", sep='\n')
    print("Female survivors:", str(sex.loc[1, 'Survived'] / num_passengers * 100) + " %\n", sep='\n')

    # 4.
    survivors_by_class = d.loc[:, ['PClass', 'Survived']].groupby('PClass').sum()
    passengers_by_class = d.loc[:, ['PClass', 'Survived']].groupby('PClass').count()

    print("Class survival statistics:")
    for c in survivors_by_class.index:
        s = survivors_by_class.loc[c, 'Survived']
        p = passengers_by_class.loc[c, 'Survived']
        print(s, 'out of', p, 'from', c if c != '*' else 'unknown', 'class survived -', s / p * 100, '%')

    # 5.
    survivors_by_decade = d.loc[:, ['Decade', 'Survived']].fillna('unknown'). \
                              groupby('Decade').sum() / num_survivors * 100
    survivors_by_decade.plot(kind='bar', figsize=(10, 10), title='Percentage of survivors based on their age'). \
        set_ylabel('Percentage')
    matplotlib.pyplot.savefig('decades.png')

    # 6.
    d.loc[:, ['Age', 'PClass']].groupby('PClass').mean().dropna(). \
        plot(kind='bar', figsize=(10, 10), legend=False, title='Mean age based on class'). \
        set_ylabel('Age')
    matplotlib.pyplot.savefig('mean.png')
    d.loc[:, ['Age', 'PClass']].groupby('PClass').max().dropna(). \
        plot(kind='bar', figsize=(10, 10), legend=False, title='Maximal age based on class'). \
        set_ylabel('Age')
    matplotlib.pyplot.savefig('max.png')
    d.loc[:, ['Age', 'PClass']].groupby('PClass').min().dropna(). \
        plot(kind='bar', figsize=(10, 10), legend=False, title='Minimal age based on class'). \
        set_ylabel('Age')
    matplotlib.pyplot.savefig('min.png')

    d.loc[:, 'PClass'] = d.loc[:, 'PClass'].dropna().apply(lambda x: pclass_map[x])
    d.loc[:, ['Age', 'PClass']].dropna().plot.hexbin(x='PClass', y='Age', gridsize=17,
                                                     title='Age distribution based on class')
    matplotlib.pyplot.savefig('age_class.png')


if __name__ == "__main__":
    main()
